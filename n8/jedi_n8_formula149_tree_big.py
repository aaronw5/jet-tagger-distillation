"""JEDI-linear jet tagger, 8 particles, 3 features: the simpler version of the simplified formula: ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 63.95% (the formula: 64.64%); same class as the formula for 93.07% of jets.  2308 leaves, depth 22.
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
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sum_pt_top5=sum(pt[:5]),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
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
    if Q.girth2 > 0.009165859781205654:
        if Q.mass > 43.67366409301758:
            if Q.sum_pt > 882.47265625:
                if Q.pt_7 > 39.984375:
                    if Q.eccentricity > 0.912938803434372:
                        if Q.sum_pt > 931.625:
                            return 'g'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.012932945508509874:
                                return 'g'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt > 0.12427913397550583:
                                    return 'g'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 74% of the training jets here get this class from the formula
                    else:
                        return 't'   # 79% of the training jets here get this class from the formula
                else:
                    if Q.mass > 132.37368774414062:
                        if Q.lam2 > 0.0002941660932265222:
                            if Q.eccentricity > 0.9305162727832794:
                                return 'q'   # 56% of the training jets here get this class from the formula
                            else:
                                return 't'   # 85% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 21.65625:
                                return 'g'   # 94% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.91094183921814:
                                    return 'g'   # 84% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 96% of the training jets here get this class from the formula
                    else:
                        if Q.z_dr_0p05_0p1 > 0.7589919865131378:
                            if Q.pt_7 > 21.7734375:
                                if Q.sum_pt > 944.1015625:
                                    if Q.planar_flow > 0.09011255577206612:
                                        return 't'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p2_0p4 > 0.12376349046826363:
                                        return 't'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.030896094627678394:
                                            return 't'   # 48% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 52% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 968.30859375:
                                    return 'g'   # 44% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 843.890625:
                                        return 'q'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 50% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top5 > 811.75:
                                if Q.planar_flow > 0.10744514688849449:
                                    return 't'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.019530246034264565:
                                        return 'g'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 112.29305267333984:
                                            if Q.pt_7 > 18.6640625:
                                                return 't'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.009538272395730019:
                                                return 't'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 48% of the training jets here get this class from the formula
                            else:
                                return 't'   # 91% of the training jets here get this class from the formula
            else:
                if Q.girth2 > 0.00967616168782115:
                    if Q.lam1 > 0.0328548289835453:
                        if Q.planar_flow > 0.03109842073172331:
                            if Q.mass > 64.53014373779297:
                                if Q.lam1 > 0.043537527322769165:
                                    if Q.e2 > 0.11295679956674576:
                                        if Q.lam1 > 0.04902449809014797:
                                            if Q.mass > 89.06569290161133:
                                                if Q.tau21 > 0.26748280227184296:
                                                    return 't'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.05751624330878258:
                                            if Q.eccentricity > 0.9852328300476074:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 5.956969261169434:
                                                    if Q.z_dr_0p2_0p4 > 0.7121904194355011:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 43% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.08720659464597702:
                                        if Q.log_sum_pt > 6.5758819580078125:
                                            if Q.lam2 > 0.00172713358188048:
                                                return 't'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 26.8359375:
                                                if Q.lam1 > 0.03824511542916298:
                                                    if Q.e2 > 0.10535428300499916:
                                                        return 't'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.19204310327768326:
                                                            if Q.planar_flow > 0.09782296419143677:
                                                                return 't'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.D2 > 0.3303734064102173:
                                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 84.09167861938477:
                                                    if Q.z_7 > 0.04020574502646923:
                                                        return 't'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 0.7475171983242035:
                                                        return 't'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.472027912735939:
                                                            return 't'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.05119170993566513:
                                            if Q.lam1 > 0.0340330321341753:
                                                if Q.tau21 > 0.10651147738099098:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 67% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 29.078125:
                                    return 't'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.28434063494205475:
                                        if Q.centroid_offset > 0.07247663661837578:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.10670441389083862:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 54% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                        else:
                            if Q.z_7 > 0.05331357195973396:
                                if Q.LHA > 0.4819241464138031:
                                    if Q.mass > 119.4515609741211:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.1964655965566635:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.041993215680122375:
                                                return 'g'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.1385338306427:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 665.0625:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 76.77167892456055:
                                            if Q.e2 > 0.08560636639595032:
                                                return 't'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.16184354573488235:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.613740682601929:
                                    return 'g'   # 68% of the training jets here get this class from the formula
                                else:
                                    if Q.width > 0.03488871082663536:
                                        return 'q'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.08933153375983238:
                                            return 't'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 62% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 51.488277435302734:
                            if Q.log_sum_pt > 6.655460357666016:
                                if Q.pt_7 > 54.953125:
                                    if Q.lam2 > 0.0011173878447152674:
                                        return 't'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.016741766594350338:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 59.0:
                                                if Q.sum_pt > 818.625:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.027077422477304935:
                                        if Q.z_7 > 0.032813536003232:
                                            if Q.tau32 > 0.44200389087200165:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.02501970250159502:
                                            if Q.girth2 > 0.010192064568400383:
                                                if Q.pt_7 > 41.203125:
                                                    if Q.centroid_offset > 0.01981825940310955:
                                                        if Q.lam2 > 0.0002193832624470815:
                                                            return 't'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 797.75:
                                                                if Q.C2 > 0.02273153979331255:
                                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt_top5 > 682.1875:
                                                                        return 'g'   # 63% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.021948279812932014:
                                                        if Q.e2 > 0.06930239871144295:
                                                            return 't'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 36% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.03172706067562103:
                                                            if Q.lam2 > 0.0001432180288247764:
                                                                return 't'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau21 > 0.10542183741927147:
                                                                    if Q.z_7 > 0.03449435532093048:
                                                                        return 't'   # 50% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.044382452964782715:
                                                    if Q.e2_sq > 0.00984952924773097:
                                                        return 't'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.019249040633440018:
                                                if Q.C2 > 0.07161491364240646:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 113.86473846435547:
                                                        return 'q'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.028551962226629257:
                                                            if Q.sum_pt_top5 > 769.875:
                                                                return 'q'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 0.00013750199286732823:
                                                                    return 't'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 20.8359375:
                                    if Q.width > 0.00992256635800004:
                                        if Q.lam1 > 0.027503905817866325:
                                            if Q.e2 > 0.07412302121520042:
                                                if Q.z_7 > 0.03736918047070503:
                                                    if Q.mass > 62.21644973754883:
                                                        if Q.log_sum_pt > 6.602492332458496:
                                                            if Q.C2 > 0.035285696387290955:
                                                                return 't'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_7 > 0.05633324943482876:
                                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 23.8203125:
                                                                if Q.lam2 > 0.00012723450345220044:
                                                                    if Q.mass > 73.71517181396484:
                                                                        return 't'   # 100% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.planar_flow > 0.0645107738673687:
                                                                            return 't'   # 99% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.girth > 0.16775131970643997:
                                                                                if Q.D2 > 0.42458055913448334:
                                                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 't'   # 92% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 114.77457046508789:
                                                                        return 't'   # 71% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.log_sum_pt > 6.066541910171509:
                                                                            return 't'   # 97% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.z_dr_0p1_0p2 > 0.616317629814148:
                                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.07009187713265419:
                                                            if Q.pt_7 > 25.9296875:
                                                                return 't'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.D2 > 0.9654340744018555:
                                                                    return 't'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth > 0.1621871441602707:
                                                                return 'g'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.126311544328928:
                                                        return 't'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 389.140625:
                                                    if Q.z_7 > 0.04542280733585358:
                                                        if Q.pt_7 > 36.296875:
                                                            if Q.lam1 > 0.029359513893723488:
                                                                return 'g'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.10185849294066429:
                                                        return 't'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.09687085077166557:
                                                            return 't'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 57.47271537780762:
                                                if Q.pt_7 > 58.640625:
                                                    if Q.e2_sq > 0.011267016641795635:
                                                        return 't'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.12202214822173119:
                                                            return 'g'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.010784208308905363:
                                                        if Q.lam2 > 0.00035330114769749343:
                                                            return 't'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 414.796875:
                                                                if Q.C2 > 0.03563763201236725:
                                                                    if Q.pt_7 > 48.953125:
                                                                        if Q.e2_sq > 0.013190639205276966:
                                                                            return 't'   # 91% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.e2 > 0.04000506177544594:
                                                                                if Q.max_dr > 0.2047310397028923:
                                                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 't'   # 64% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 94% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.z_7 > 0.033114176243543625:
                                                                            if Q.LHA > 0.43350090086460114:
                                                                                if Q.e2 > 0.06703386828303337:
                                                                                    return 't'   # 99% of the training jets here get this class from the formula
                                                                                else:
                                                                                    if Q.sum_pt_top5 > 421.390625:
                                                                                        return 't'   # 76% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 98% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.07510148361325264:
                                                                    return 't'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam1 > 0.024790734983980656:
                                                                        return 'g'   # 65% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.2607443779706955:
                                                            return 't'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.046591516584157944:
                                                                if Q.eccentricity > 0.9575228989124298:
                                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.girth > 0.09474015608429909:
                                                                        return 't'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.024350764229893684:
                                                    if Q.planar_flow > 0.10282062366604805:
                                                        return 't'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.04124535992741585:
                                                            return 'g'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.0007959330105222762:
                                                        return 't'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.047996366396546364:
                                                            if Q.C2 > 0.07208935916423798:
                                                                return 't'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p2_0p4 > 0.1032387763261795:
                                                                    if Q.pt_7 > 33.4375:
                                                                        if Q.lam2 > 0.0002460831601638347:
                                                                            if Q.mass > 54.690093994140625:
                                                                                return 't'   # 74% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'g'   # 74% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.pt_7 > 26.8828125:
                                                                            return 't'   # 83% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 60% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.tau21 > 0.25934717059135437:
                                                                        if Q.pt_7 > 40.109375:
                                                                            return 'g'   # 53% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 83% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.sum_pt_top5 > 342.515625:
                                                                            return 't'   # 98% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.lam2 > 0.0004268841730663553:
                                                                                return 't'   # 97% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'g'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.3347530663013458:
                                            if Q.mass > 68.65746688842773:
                                                if Q.e2 > 0.04658855311572552:
                                                    if Q.z_7 > 0.051807695999741554:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.043308403342962265:
                                                if Q.mass > 55.92749786376953:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.02307417243719101:
                                        if Q.z_7 > 0.029372558929026127:
                                            if Q.mass > 68.47557830810547:
                                                return 't'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.30419452488422394:
                                                return 'q'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p05_0p1 > 0.8868470191955566:
                                            return 't'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.024715530686080456:
                                                return 't'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.026883521117269993:
                                                    if Q.planar_flow > 0.10217864438891411:
                                                        return 't'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 96% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 447.0234375:
                                if Q.C2 > 0.04458998143672943:
                                    if Q.z_7 > 0.0733010545372963:
                                        if Q.centroid_offset > 0.04139614477753639:
                                            return 't'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.000639956968370825:
                                                if Q.C2 > 0.08443953096866608:
                                                    return 't'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.32045793533325195:
                                                        return 't'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p05_0p1 > 0.22018884122371674:
                                                    if Q.sum_pt > 493.3125:
                                                        return 't'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.10524418577551842:
                                            if Q.planar_flow > 0.3549595922231674:
                                                return 't'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 1.5032005310058594:
                                                    return 't'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 96% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.lam2 > 0.0013343816390261054:
                                    if Q.lam1 > 0.0246749147772789:
                                        if Q.pt_7 > 24.6171875:
                                            return 't'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.01310695894062519:
                                            return 't'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 34.984375:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.03898690268397331:
                                        if Q.C2 > 0.09109079837799072:
                                            return 't'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.00032378581818193197:
                                                if Q.log_sum_pt > 5.801953077316284:
                                                    if Q.z_dr_0p05_0p1 > 0.2638000398874283:
                                                        if Q.mass > 46.036699295043945:
                                                            if Q.tau21 > 0.2218215987086296:
                                                                if Q.C2 > 0.07362165302038193:
                                                                    return 't'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.max_dr > 0.1999511495232582:
                                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam1 > 0.010754191782325506:
                                                                if Q.planar_flow > 0.1645350307226181:
                                                                    if Q.e2 > 0.056341882795095444:
                                                                        return 't'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p05_0p1 > 0.09826619550585747:
                                                    if Q.max_dr > 0.20455703884363174:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_0 > 74.8125:
                                                            if Q.z_dr_0p05_0p1 > 0.28851063549518585:
                                                                return 'g'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2_sq > 0.016899376176297665:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p1_0p2 > 0.6251974105834961:
                                                            return 't'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 335.546875:
                                            if Q.max_dr > 0.20116189867258072:
                                                if Q.planar_flow > 0.08209586143493652:
                                                    return 't'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.04424457065761089:
                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.3504764437675476:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 93% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.04596173018217087:
                        if Q.mass > 59.217830657958984:
                            if Q.pt_7 > 29.890625:
                                if Q.width > 0.009498610626906157:
                                    if Q.tau32 > 0.4401197284460068:
                                        if Q.mass > 63.62969398498535:
                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 73% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 73.98112869262695:
                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.48690687119960785:
                                        return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 50% of the training jets here get this class from the formula
                        else:
                            if Q.tau32 > 0.426859512925148:
                                if Q.width > 0.009336075279861689:
                                    return 't'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 36.421875:
                                        if Q.mass > 50.215871810913086:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.08499637991189957:
                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 34.53125:
                                    if Q.centroid_offset > 0.027334737591445446:
                                        return 't'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 52.53964614868164:
                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 92% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 73.9455680847168:
                            if Q.max_dr > 0.24327897280454636:
                                return 't'   # 77% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 82% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.041300300508737564:
                                if Q.mass > 61.32402992248535:
                                    if Q.pt_7 > 33.734375:
                                        if Q.centroid_offset > 0.023488761857151985:
                                            return 't'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p1_0p2 > 0.09651458263397217:
                                        if Q.tau32 > 0.3916410058736801:
                                            return 't'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.1797238290309906:
                                                return 't'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 42% of the training jets here get this class from the formula
                            else:
                                return 't'   # 96% of the training jets here get this class from the formula
        else:
            if Q.sum_pt > 414.2109375:
                if Q.centroid_offset > 0.05281537212431431:
                    if Q.girth2 > 0.009838216938078403:
                        return 't'   # 97% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 478.640625:
                            return 't'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.028159678913652897:
                                return 'g'   # 60% of the training jets here get this class from the formula
                            else:
                                return 't'   # 86% of the training jets here get this class from the formula
                else:
                    if Q.tau21 > 0.22486918419599533:
                        if Q.z_7 > 0.07759921997785568:
                            return 'g'   # 75% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 28.3359375:
                                if Q.LHA > 0.29485245048999786:
                                    if Q.z_7 > 0.07477852329611778:
                                        return 'g'   # 50% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 85% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 64% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.09364914894104:
                                    return 't'   # 69% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 77% of the training jets here get this class from the formula
                    else:
                        return 't'   # 91% of the training jets here get this class from the formula
            else:
                if Q.centroid_offset > 0.061102135106921196:
                    if Q.log_sum_pt > 5.776466131210327:
                        if Q.girth2 > 0.010715829208493233:
                            if Q.centroid_offset > 0.07148662582039833:
                                return 't'   # 94% of the training jets here get this class from the formula
                            else:
                                if Q.planar_flow > 0.08839289844036102:
                                    if Q.mass > 36.46353340148926:
                                        return 't'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.2270980328321457:
                                            return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 60% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 66% of the training jets here get this class from the formula
                    else:
                        if Q.width > 0.013486417941749096:
                            if Q.log_sum_pt > 5.642294883728027:
                                if Q.planar_flow > 0.4507753700017929:
                                    return 't'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_0 > 56.4375:
                                        return 'g'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 71% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 68% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 95% of the training jets here get this class from the formula
                else:
                    if Q.mass > 38.103023529052734:
                        if Q.lam2 > 0.001470613176934421:
                            if Q.girth2 > 0.013151093851774931:
                                if Q.pt_7 > 23.6328125:
                                    if Q.lam2 > 0.003501403727568686:
                                        return 't'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.2509751543402672:
                                            if Q.D2 > 1.3229411840438843:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 41.40318298339844:
                                                    return 't'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 98% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 68% of the training jets here get this class from the formula
                            else:
                                if Q.tau21 > 0.3202938884496689:
                                    if Q.pt_7 > 31.96875:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 51% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 84% of the training jets here get this class from the formula
                        else:
                            if Q.C2 > 0.033211635425686836:
                                if Q.C2 > 0.10287746414542198:
                                    return 't'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 287.21875:
                                        if Q.girth2 > 0.012205907143652439:
                                            if Q.sum_pt_top5 > 296.765625:
                                                return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.planar_flow > 0.09395861253142357:
                                    return 't'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 398.4921875:
                                        return 't'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt_top5 > 281.109375:
                            if Q.tau21 > 0.15028869360685349:
                                if Q.D2 > 2.200949549674988:
                                    return 't'   # 52% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 81% of the training jets here get this class from the formula
                            else:
                                return 't'   # 53% of the training jets here get this class from the formula
                        else:
                            if Q.lam2 > 0.004930727416649461:
                                return 'g'   # 64% of the training jets here get this class from the formula
                            else:
                                if Q.tau21 > 0.19738037139177322:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.eccentricity > 0.8838045597076416:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 60% of the training jets here get this class from the formula
    else:
        if Q.mass > 29.730234146118164:
            if Q.width > 0.0066669650841504335:
                if Q.centroid_offset > 0.03028692863881588:
                    if Q.centroid_offset > 0.037452783435583115:
                        if Q.sum_pt > 466.3984375:
                            if Q.sum_pt > 775.203125:
                                if Q.pt_7 > 28.265625:
                                    if Q.sum_pt > 865.578125:
                                        return 'g'   # 66% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 33.78125:
                                            if Q.lam2 > 0.00032073848706204444:
                                                return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.00783567689359188:
                                                    return 't'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 1.8096078038215637:
                                        return 'q'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 59% of the training jets here get this class from the formula
                            else:
                                if Q.tau21 > 0.2631501257419586:
                                    if Q.z_dr_0p1_0p2 > 0.29703132808208466:
                                        if Q.mass > 37.943899154663086:
                                            return 't'   # 46% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 35.546875:
                                            if Q.centroid_offset > 0.04294038377702236:
                                                if Q.e2 > 0.0338053684681654:
                                                    return 'g'   # 43% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 1.9004546403884888:
                                                    return 't'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.00010129139627679251:
                                                return 't'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 624.46875:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.510537147521973:
                                        if Q.lam2 > 0.00018224670930067077:
                                            return 't'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.007645276375114918:
                                                return 't'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 32.671875:
                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.tau21 > 0.22161612659692764:
                                if Q.z_dr_0p05_0p1 > 0.8168047368526459:
                                    if Q.log_sum_pt > 6.096439838409424:
                                        return 't'   # 69% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 73% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.z_dr_0p1_0p2 > 0.27116838097572327:
                                    if Q.mass > 33.0008602142334:
                                        return 't'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 89% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 85% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 34.578125:
                            if Q.width > 0.008150397799909115:
                                if Q.tau32 > 0.49346086382865906:
                                    if Q.sum_pt > 707.78125:
                                        return 'Z'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.12022032961249352:
                                            return 't'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.09276754409074783:
                                                return 't'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.03531361371278763:
                                        if Q.mass > 42.82220268249512:
                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 50% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 530.859375:
                                    if Q.LHA > 0.3090876489877701:
                                        if Q.lam2 > 0.00020861926168436185:
                                            if Q.C2 > 0.024298114702105522:
                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0002054741999018006:
                                            if Q.tau21 > 0.23397858440876007:
                                                return 'Z'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.495265483856201:
                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.5514936447143555:
                                                    return 't'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.16459842771291733:
                                        if Q.mass > 34.56219482421875:
                                            if Q.girth > 0.07601579651236534:
                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.03232656978070736:
                                            return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 54% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 34.449262619018555:
                                if Q.log_sum_pt > 6.6203649044036865:
                                    if Q.lam2 > 6.961002873140387e-05:
                                        if Q.log_sum_pt > 6.689965486526489:
                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.19910567998886108:
                                                return 't'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 39% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 551.0625:
                                        if Q.max_dr > 0.15020927786827087:
                                            if Q.tau21 > 0.25063127279281616:
                                                if Q.C2 > 0.07161770761013031:
                                                    return 't'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.32576583325862885:
                                                        return 't'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 27.953125:
                                                if Q.z_dr_0p1_0p2 > 0.21328487992286682:
                                                    if Q.centroid_offset > 0.03290167450904846:
                                                        return 't'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.02288345154374838:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 95% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.042298793792725:
                                    return 't'   # 54% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 90% of the training jets here get this class from the formula
                else:
                    if Q.max_dr > 0.2370653823018074:
                        if Q.log_sum_pt > 6.62684178352356:
                            if Q.width > 0.00834614410996437:
                                if Q.max_dr > 0.3012365847826004:
                                    if Q.tau21 > 0.1310022696852684:
                                        if Q.D2 > 2.7914661169052124:
                                            return 't'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 72.27254867553711:
                                        if Q.mass > 85.57328796386719:
                                            return 't'   # 48% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 67% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 90.9301872253418:
                                    return 't'   # 77% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 3.9318548440933228:
                                        return 't'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.00030507218616548926:
                                            if Q.tau32 > 0.34742991626262665:
                                                return 't'   # 49% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 98% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 33.078125:
                                if Q.width > 0.008280016016215086:
                                    if Q.e2 > 0.036362623795866966:
                                        if Q.pt_0 > 118.40625:
                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.488111972808838:
                                        if Q.max_dr > 0.31807655096054077:
                                            return 't'   # 55% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.4038337767124176:
                                            if Q.lam2 > 0.00018910384824266657:
                                                return 't'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.308609485626221:
                                                    if Q.tau21 > 0.1349114030599594:
                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 83% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.111449241638184:
                                    if Q.max_dr > 0.28246164321899414:
                                        return 't'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.007907360792160034:
                                            return 't'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.508524417877197:
                                                if Q.centroid_offset > 0.016339515335857868:
                                                    return 't'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 36.18799018859863:
                                        return 't'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 79% of the training jets here get this class from the formula
                    else:
                        if Q.girth2 > 0.006853858241811395:
                            if Q.sum_pt > 509.921875:
                                if Q.centroid_offset > 0.023243707604706287:
                                    if Q.pt_7 > 28.9765625:
                                        if Q.width > 0.008406830951571465:
                                            if Q.e2 > 0.043069420382380486:
                                                if Q.sum_pt > 586.40625:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.5399837493896484:
                                                        if Q.pt_7 > 36.40625:
                                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.480833530426025:
                                                    if Q.eccentricity > 0.9761360585689545:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.5501330494880676:
                                                        return 't'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p05_0p1 > 0.07992958649992943:
                                                if Q.lam2 > 0.00024658114125486463:
                                                    if Q.tau32 > 0.5649907886981964:
                                                        if Q.LHA > 0.3169845938682556:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 561.5625:
                                                        return 'Z'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.11517377197742462:
                                                            if Q.tau21 > 0.10791027918457985:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.04113227687776089:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.504428386688232:
                                            if Q.girth2 > 0.007976073771715164:
                                                if Q.e2 > 0.04174797236919403:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 26.25:
                                                if Q.tau21 > 0.22350571304559708:
                                                    return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.1341845691204071:
                                                        return 't'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.00012232514200150035:
                                                            return 't'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1086.34765625:
                                        if Q.pt_7 > 43.515625:
                                            return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.16425975412130356:
                                            if Q.sum_pt > 701.40234375:
                                                if Q.girth2 > 0.009014691691845655:
                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 749.4375:
                                                        return 'Z'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 24.4921875:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_over_sum_pt > 0.08645743504166603:
                                                                return 't'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 27.3359375:
                                                    if Q.width > 0.008325271774083376:
                                                        if Q.tau32 > 0.453179195523262:
                                                            if Q.pt_7 > 35.859375:
                                                                if Q.width > 0.009013717994093895:
                                                                    return 't'   # 66% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0p05_0p1 > 0.23144032061100006:
                                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.pt_7 > 43.453125:
                                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 29.796875:
                                                            if Q.sum_pt > 580.71875:
                                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau21 > 0.1270110011100769:
                                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.011819561012089252:
                                                                        return 't'   # 65% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p05_0p1 > 0.4565845727920532:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.007315129041671753:
                                                                    return 't'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.3971671611070633:
                                                        return 't'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 51% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.006982035934925079:
                                                if Q.girth2 > 0.008827462326735258:
                                                    if Q.pt_7 > 29.1171875:
                                                        if Q.mass > 53.51461982727051:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 34.984375:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau32 > 0.47961296141147614:
                                                                    return 't'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 677.046875:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 30.5546875:
                                                        return 'Z'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.478161573410034:
                                                            if Q.sum_pt > 694.671875:
                                                                return 'Z'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_7 > 0.03286162577569485:
                                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 25.1484375:
                                                                if Q.max_dr > 0.13104406744241714:
                                                                    if Q.tau32 > 0.5372945666313171:
                                                                        if Q.log_sum_pt > 6.38574743270874:
                                                                            if Q.z_7 > 0.045455701649188995:
                                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 57% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 75% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.04399280995130539:
                                                    if Q.centroid_offset > 0.006659358972683549:
                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p05_0p1 > 0.6305217742919922:
                                                            if Q.max_dr > 0.10294293612241745:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.2835733890533447:
                                                        if Q.e2 > 0.042303672060370445:
                                                            if Q.centroid_offset > 0.005131760844960809:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 59.54317283630371:
                                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_7 > 40.5:
                                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.14644617587327957:
                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.012310393620282412:
                                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 60% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 32.546875:
                                    if Q.log_sum_pt > 5.981019735336304:
                                        if Q.girth2 > 0.008906284812837839:
                                            if Q.tau32 > 0.3952035456895828:
                                                if Q.C2 > 0.04597778804600239:
                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 38.546875:
                                                        if Q.e2 > 0.048260968178510666:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.07835793122649193:
                                                if Q.e2 > 0.038049664348363876:
                                                    if Q.width > 0.006927129812538624:
                                                        if Q.girth2 > 0.008350378833711147:
                                                            if Q.e2 > 0.044664137065410614:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.010823372285813093:
                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.20740847289562225:
                                                    if Q.log_sum_pt > 6.093252182006836:
                                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.4355827420949936:
                                                            if Q.D2 > 1.1849704384803772:
                                                                return 'g'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.08086733892560005:
                                                        if Q.lam2 > 0.0002206845674663782:
                                                            return 't'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.02312440425157547:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 49% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.14000660181045532:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.008236670400947332:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 32.25016403198242:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 5.970244884490967:
                                        if Q.pt_7 > 29.7421875:
                                            if Q.e2 > 0.04446675814688206:
                                                if Q.width > 0.008370983880013227:
                                                    if Q.tau32 > 0.4830922335386276:
                                                        return 't'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt > 0.08541061729192734:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p05_0p1 > 0.6381310224533081:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.23696377128362656:
                                                    if Q.eccentricity > 0.9520993530750275:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 433.7734375:
                                                if Q.LHA > 0.2797044813632965:
                                                    if Q.tau32 > 0.3892579823732376:
                                                        return 't'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 28.375:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 26.2265625:
                                                    return 't'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.040887435898184776:
                                if Q.centroid_offset > 0.013332781847566366:
                                    if Q.z_dr_0p1_0p2 > 0.33429740369319916:
                                        if Q.girth2 > 0.00674959784373641:
                                            return 'Z'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 34.44139099121094:
                                            if Q.width > 0.006747415056452155:
                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 0.9203856587409973:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 691.90625:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.016216753982007504:
                                                            if Q.e2 > 0.0425485298037529:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 31% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 789.953125:
                                        if Q.girth2 > 0.0067559340968728065:
                                            if Q.max_dr > 0.0940076969563961:
                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.005044342251494527:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 72.95259857177734:
                                                return 'Z'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.10150958597660065:
                                                    if Q.centroid_offset > 0.005491289310157299:
                                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.007054156623780727:
                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.0067973576951771975:
                                            if Q.centroid_offset > 0.005462405038997531:
                                                if Q.log_sum_pt > 6.567067861557007:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 42.59375:
                                                        if Q.girth2 > 0.006815186934545636:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.13058383017778397:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.007653353037312627:
                                                                if Q.pt_0 > 161.5:
                                                                    return 'Z'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.10184852406382561:
                                                    if Q.z_dr_0p05_0p1 > 0.5272465348243713:
                                                        if Q.sum_pt > 698.671875:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.1154671274125576:
                                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.0727006793022156:
                                                if Q.planar_flow > 0.27219705283641815:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 44% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.011104677338153124:
                                                    if Q.girth2 > 0.006753571797162294:
                                                        if Q.mass > 52.77418327331543:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 43.046875:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.11514914408326149:
                                                        if Q.girth > 0.07739502936601639:
                                                            if Q.mass > 54.02579307556152:
                                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 676.890625:
                                    if Q.z_dr_0p1_0p2 > 0.25569669902324677:
                                        if Q.z_dr_0p05_0p1 > 0.3902977854013443:
                                            if Q.e2 > 0.039664093405008316:
                                                if Q.girth2 > 0.0067276975605636835:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.6408281326293945:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 0.4497168958187103:
                                                return 'Z'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.011852611787617207:
                                        if Q.pt_7 > 27.3828125:
                                            if Q.z_dr_0p05_0p1 > 0.14270839095115662:
                                                if Q.mass > 39.86901664733887:
                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 34.90625:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.03441622108221054:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 49.414201736450195:
                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.16609498858451843:
                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p05_0p1 > 0.5113582611083984:
                                                if Q.max_dr > 0.12278284877538681:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 76% of the training jets here get this class from the formula
            else:
                if Q.max_dr > 0.1599615514278412:
                    if Q.girth2 > 0.004335511475801468:
                        if Q.centroid_offset > 0.012019566260278225:
                            if Q.mass > 40.1126708984375:
                                if Q.lam1 > 0.00501412944868207:
                                    if Q.D2 > 3.317728877067566:
                                        if Q.sum_pt > 756.375:
                                            if Q.centroid_offset > 0.03459158353507519:
                                                if Q.pt_7 > 25.296875:
                                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.028009409084916115:
                                                if Q.pt_7 > 25.7734375:
                                                    if Q.eccentricity > 0.9734203219413757:
                                                        return 'Z'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 718.9375:
                                            if Q.log_sum_pt > 6.965810298919678:
                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.2059929072856903:
                                                    if Q.max_dr > 0.17352253198623657:
                                                        return 'Z'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.26631373167037964:
                                                            if Q.centroid_offset > 0.013496457133442163:
                                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.005680728005245328:
                                                                    return 'Z'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.014812381938099861:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.16659394651651382:
                                                                    if Q.lam1 > 0.005286500556394458:
                                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 28.1953125:
                                                if Q.LHA > 0.22340166568756104:
                                                    if Q.centroid_offset > 0.03990001417696476:
                                                        if Q.C2 > 0.04273918829858303:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.014911665115505457:
                                                            if Q.eccentricity > 0.887509673833847:
                                                                if Q.e2 > 0.03586771711707115:
                                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt > 611.03125:
                                                                        return 'Z'   # 99% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.tau21 > 0.1324765831232071:
                                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.centroid_offset > 0.018764222040772438:
                                                                                if Q.girth2 > 0.006268950179219246:
                                                                                    return 't'   # 54% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.027885079383850098:
                                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.005514618940651417:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.17538641393184662:
                                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.48142118752002716:
                                                        if Q.pt_7 > 37.953125:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.2513020634651184:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.4470953941345215:
                                                        if Q.pt_7 > 23.3984375:
                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.20483365654945374:
                                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.04485522769391537:
                                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p1_0p2 > 0.1052197702229023:
                                        if Q.centroid_offset > 0.01704360730946064:
                                            if Q.girth2 > 0.004551665624603629:
                                                if Q.max_dr > 0.16418170183897018:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.021248837001621723:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.004730543354526162:
                                                            return 'Z'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.01928937155753374:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.004428677726536989:
                                                        return 'Z'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.18103761225938797:
                                                if Q.girth > 0.05245758220553398:
                                                    if Q.girth2 > 0.004923333413898945:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p1_0p2 > 0.11497117951512337:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.11956336349248886:
                                                        return 'Z'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 62.41113090515137:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.0007005619700066745:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.015555084683001041:
                                                            if Q.max_dr > 0.1694284826517105:
                                                                if Q.lam1 > 0.004635275807231665:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.026160601526498795:
                                            if Q.girth2 > 0.00490059657022357:
                                                if Q.D2 > 1.7449126243591309:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.005437960848212242:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.033524466678500175:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 55% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.01774340122938156:
                                                    if Q.e2 > 0.029375911690294743:
                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.004619056824594736:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.005317723145708442:
                                                if Q.e2 > 0.022134464234113693:
                                                    if Q.log_sum_pt > 6.56144642829895:
                                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.014154368545860052:
                                                    if Q.D2 > 4.236284971237183:
                                                        if Q.sum_pt > 772.203125:
                                                            if Q.centroid_offset > 0.03425918519496918:
                                                                return 'q'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.024816050194203854:
                                                                return 'q'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p2_0p4 > 0.08370860293507576:
                                                            return 'Z'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.015186124946922064:
                                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.18893616646528244:
                                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.21550806611776352:
                                                        if Q.mass > 48.939300537109375:
                                                            if Q.max_dr > 0.22715377807617188:
                                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.004510874627158046:
                                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.004747630562633276:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 59.2119026184082:
                                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 32.109375:
                                    if Q.centroid_offset > 0.01904193125665188:
                                        if Q.width > 0.006018914747983217:
                                            if Q.e2 > 0.028122839517891407:
                                                if Q.log_sum_pt > 6.131222486495972:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.2278851494193077:
                                                if Q.lam2 > 0.0006563659990206361:
                                                    if Q.tau21 > 0.2579289674758911:
                                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 31.432852745056152:
                                                        if Q.e2 > 0.025719855912029743:
                                                            if Q.width > 0.00494187162257731:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.1706719771027565:
                                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 42% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.1783502846956253:
                                            if Q.mass > 33.74386215209961:
                                                if Q.girth2 > 0.004969214787706733:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.13755590468645096:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.00582831003703177:
                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.228788375854492:
                                        if Q.pt_7 > 26.8359375:
                                            if Q.girth2 > 0.0058924739714711905:
                                                return 't'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.01981380768120289:
                                                    if Q.tau21 > 0.2225332260131836:
                                                        if Q.centroid_offset > 0.04075218364596367:
                                                            return 't'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 35.730960845947266:
                                                            return 'Z'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.9269101023674011:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 615.9140625:
                                                return 'q'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.004773157648742199:
                                                    return 't'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 25.3046875:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 33.96136665344238:
                                            return 't'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.2207399234175682:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 66% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.005514880875125527:
                                if Q.max_dr > 0.1796054244041443:
                                    if Q.mass > 54.47787094116211:
                                        if Q.width > 0.005850579822435975:
                                            if Q.C2 > 0.09405714273452759:
                                                return 't'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 85.45046615600586:
                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 62.100263595581055:
                                                        return 'Z'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.32136011123657227:
                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 24.203125:
                                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 755.1796875:
                                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 0.7575982511043549:
                                                if Q.e2 > 0.027774719521403313:
                                                    if Q.centroid_offset > 0.005320520373061299:
                                                        if Q.lam1 > 0.005525484215468168:
                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.007925761397928:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.029349450953304768:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 3.5906111001968384:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 1.0308208465576172:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.006337226368486881:
                                                                return 'Z'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 881.78125:
                                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.00651993160136044:
                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 73.84429931640625:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 27.90625:
                                            if Q.max_dr > 0.20651522278785706:
                                                if Q.log_sum_pt > 6.187975168228149:
                                                    if Q.pt_7 > 36.609375:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.21337366849184036:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 49% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.006057626800611615:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.008330141194164753:
                                                        if Q.e2 > 0.03288022801280022:
                                                            return 'W'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.006060967221856117:
                                        if Q.centroid_offset > 0.006446602288633585:
                                            if Q.mass > 57.822988510131836:
                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.19252392649650574:
                                                    if Q.D2 > 0.8162441849708557:
                                                        return 'Z'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 29.7734375:
                                                        if Q.lam1 > 0.006281048292294145:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 48.3087158203125:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 47% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 42% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 64.20256805419922:
                                                if Q.width > 0.006349456030875444:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.792020082473755:
                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.16946010291576385:
                                                    if Q.pt_0 > 200.625:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.00799795938655734:
                                            if Q.max_dr > 0.1686018854379654:
                                                if Q.mass > 53.40480041503906:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.005866829305887222:
                                                    if Q.log_sum_pt > 6.635331153869629:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 77.30388641357422:
                                                return 'Z'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.17635401338338852:
                                                    if Q.mass > 66.08983993530273:
                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.22798065841197968:
                                    if Q.max_dr > 0.26231835782527924:
                                        if Q.mass > 54.32900810241699:
                                            if Q.lam1 > 0.004676132462918758:
                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 2.5022835731506348:
                                                    if Q.e2 > 0.01872286293655634:
                                                        if Q.girth2 > 0.004605212481692433:
                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.008572013583034277:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 30.4453125:
                                                if Q.girth > 0.03732862323522568:
                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.0044891019351780415:
                                                    return 't'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 40% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.004868061980232596:
                                            if Q.centroid_offset > 0.006079554324969649:
                                                if Q.LHA > 0.21409239619970322:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.24261190742254257:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 48% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2_sq > 0.00538426311686635:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 66.8561019897461:
                                                        if Q.pt_7 > 16.4453125:
                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.0109695540741086:
                                                if Q.lam1 > 0.004494855646044016:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 51% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 946.0859375:
                                                    if Q.z_7 > 0.023420626297593117:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.007502814754843712:
                                                        if Q.max_dr > 0.2461947798728943:
                                                            if Q.width > 0.004599221749231219:
                                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 66.41122436523438:
                                        if Q.z_dr_0p1_0p2 > 0.13276614248752594:
                                            if Q.mass > 74.68409729003906:
                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 34.515625:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 25.546875:
                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.009131823666393757:
                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 72.20516204833984:
                                                        if Q.pt_7 > 15.48828125:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.005119451321661472:
                                                            if Q.mass > 68.21406936645508:
                                                                if Q.z_7 > 0.01669558882713318:
                                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.1968730315566063:
                                            if Q.width > 0.004993900191038847:
                                                if Q.centroid_offset > 0.008874977473169565:
                                                    if Q.mass > 54.81357955932617:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 35.171875:
                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.005080832168459892:
                                                        if Q.max_dr > 0.21312449872493744:
                                                            if Q.girth2 > 0.005318013485521078:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.036648087203502655:
                                                                    return 'Z'   # 46% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 32.22418022155762:
                                                if Q.girth2 > 0.005190621828660369:
                                                    if Q.centroid_offset > 0.009936691727489233:
                                                        if Q.max_dr > 0.1717602014541626:
                                                            if Q.e2 > 0.028782251290977:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.log_sum_pt > 6.718048572540283:
                                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 59% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.021438544616103172:
                            if Q.lam1 > 0.0030065097380429506:
                                if Q.max_dr > 0.1724618747830391:
                                    if Q.eccentricity > 0.9746353030204773:
                                        if Q.mass > 33.75918769836426:
                                            if Q.max_dr > 0.18282094597816467:
                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.2332949936389923:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.023356467485427856:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 25.6015625:
                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 49% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.02371847815811634:
                                            if Q.LHA > 0.220168836414814:
                                                if Q.e2 > 0.01245501497760415:
                                                    if Q.width > 0.003287656349129975:
                                                        if Q.sum_pt > 670.9609375:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 25.7734375:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 40% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.003856435068883002:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.21399618685245514:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.026249583810567856:
                                        if Q.LHA > 0.24063002318143845:
                                            if Q.lam1 > 0.003520707366988063:
                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.030430853366851807:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 37.328125:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.16448774933815002:
                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.02791842445731163:
                                                return 'Z'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.004012326477095485:
                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.0035995907383039594:
                                                if Q.centroid_offset > 0.023702874779701233:
                                                    if Q.max_dr > 0.16568028181791306:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.2029530256986618:
                                    if Q.girth2 > 0.0023782020434737206:
                                        if Q.centroid_offset > 0.02384944912046194:
                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.22861461341381073:
                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2_sq > 0.002251046826131642:
                                                    if Q.mass > 40.03408241271973:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.025281733833253384:
                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.23610906302928925:
                                                if Q.lam1 > 0.002106902189552784:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.027869464829564095:
                                        if Q.max_dr > 0.17438049614429474:
                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.030936425551772118:
                                                return 'Z'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0004120374651392922:
                                            if Q.D2 > 2.7518733739852905:
                                                if Q.mass > 35.71964454650879:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 933.2890625:
                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.025212141685187817:
                                                    if Q.max_dr > 0.17983104288578033:
                                                        if Q.girth > 0.040350690484046936:
                                                            if Q.mass > 35.327125549316406:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 96% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.0017226602067239583:
                                if Q.max_dr > 0.22609303891658783:
                                    if Q.centroid_offset > 0.01511643873527646:
                                        if Q.lam1 > 0.0030092053348198533:
                                            if Q.lam1 > 0.0034015775891020894:
                                                if Q.sum_pt > 637.8671875:
                                                    if Q.centroid_offset > 0.016460705548524857:
                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.24192755669355392:
                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_over_sum_pt > 0.059623461216688156:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 38% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.2505025565624237:
                                                    if Q.eccentricity > 0.969428151845932:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_0 > 363.375:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 32% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.018351461738348007:
                                                        if Q.eccentricity > 0.9705451726913452:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 735.328125:
                                                            if Q.centroid_offset > 0.01639739330857992:
                                                                if Q.max_dr > 0.2370806336402893:
                                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 36% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.002471424057148397:
                                                if Q.D2 > 5.1776041984558105:
                                                    if Q.eccentricity > 0.9742206931114197:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.00021755863417638466:
                                                            return 'Z'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.019524939358234406:
                                                        if Q.mass > 41.51844024658203:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 45.96748352050781:
                                                            if Q.pt_7 > 17.9765625:
                                                                return 'Z'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.017378612421453:
                                                                if Q.max_dr > 0.25051039457321167:
                                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.654232501983643:
                                                    if Q.mass > 44.70308303833008:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.01865619793534279:
                                                            if Q.mass > 40.32709884643555:
                                                                return 'Z'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 54.258514404296875:
                                            if Q.centroid_offset > 0.012651104014366865:
                                                if Q.max_dr > 0.2421782687306404:
                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 62.52808952331543:
                                                    if Q.pt_7 > 11.47265625:
                                                        if Q.centroid_offset > 0.004957176046445966:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 1045.7109375:
                                                                return 'Z'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 3.7009475231170654:
                                                            return 'Z'   # 40% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.3049912750720978:
                                                        if Q.centroid_offset > 0.006338344654068351:
                                                            if Q.lam1 > 0.0038161930860951543:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_7 > 0.017928648740053177:
                                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.009606556966900826:
                                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 20.6640625:
                                                                if Q.mass > 57.01710891723633:
                                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 27.4375:
                                                            if Q.sum_pt_top5 > 802.40625:
                                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.995150566101074:
                                                                return 'W'   # 48% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.00959980022162199:
                                                                    if Q.D2 > 2.4521751403808594:
                                                                        if Q.e2_sq > 0.003851430374197662:
                                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 99% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.018254845403134823:
                                                if Q.mass > 36.03524208068848:
                                                    if Q.girth2 > 0.0035167041933164:
                                                        if Q.centroid_offset > 0.013059128541499376:
                                                            if Q.max_dr > 0.25503382086753845:
                                                                if Q.max_dr > 0.2809591591358185:
                                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam1 > 0.003723767353221774:
                                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 48% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2 > 0.00411214679479599:
                                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.3074738532304764:
                                                                if Q.log_sum_pt > 6.665146112442017:
                                                                    if Q.centroid_offset > 0.009756611194461584:
                                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 636.5625:
                                                                    if Q.max_dr > 0.28774115443229675:
                                                                        if Q.girth > 0.03385854698717594:
                                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 724.515625:
                                                            if Q.sum_pt > 1094.734375:
                                                                return 'g'   # 45% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.LHA > 0.15064968913793564:
                                                                    if Q.max_dr > 0.3312530368566513:
                                                                        if Q.girth > 0.02733925823122263:
                                                                            if Q.tau21 > 0.2848525494337082:
                                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.sum_pt > 1032.5703125:
                                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.mass > 48.250675201416016:
                                                                                if Q.centroid_offset > 0.013634335715323687:
                                                                                    if Q.max_dr > 0.2791988104581833:
                                                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                                            else:
                                                                                if Q.centroid_offset > 0.008494215551763773:
                                                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                                                                else:
                                                                                    if Q.planar_flow > 0.07477842271327972:
                                                                                        if Q.girth2 > 0.002579220221377909:
                                                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                                                        else:
                                                                                            return 'q'   # 67% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 41.3314208984375:
                                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.008174808230251074:
                                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p2_0p4 > 0.04274986498057842:
                                                                return 'W'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 39% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 640.25:
                                                        if Q.centroid_offset > 0.012333436869084835:
                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 42.62878227233887:
                                                    if Q.lam2 > 4.885714042757172e-05:
                                                        if Q.tau21 > 0.38047944009304047:
                                                            return 'W'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.18090250343084335:
                                        if Q.log_sum_pt > 6.94257926940918:
                                            if Q.pt_7 > 23.3359375:
                                                if Q.lam1 > 0.0027423594146966934:
                                                    if Q.eccentricity > 0.9864217936992645:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 33.53125:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 42% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 1129.21875:
                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.016900934278964996:
                                                if Q.girth2 > 0.003597399452701211:
                                                    if Q.max_dr > 0.17690210789442062:
                                                        if Q.planar_flow > 0.09808430448174477:
                                                            if Q.D2 > 2.5240917205810547:
                                                                if Q.girth2 > 0.0038614204386249185:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam1 > 0.0039027222665026784:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.019224364310503006:
                                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0p1_0p2 > 0.0679931566119194:
                                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.017540156841278076:
                                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.004117512376978993:
                                                            if Q.centroid_offset > 0.018830875866115093:
                                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 46.235130310058594:
                                                        if Q.z_dr_0p1_0p2 > 0.02903818618506193:
                                                            if Q.mass > 50.38421440124512:
                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.018896483816206455:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.0034588288981467485:
                                                            if Q.z_dr_0p1_0p2 > 0.06602213531732559:
                                                                return 'W'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 33.18853187561035:
                                                    if Q.centroid_offset > 0.01464900467544794:
                                                        if Q.mass > 55.473894119262695:
                                                            if Q.LHA > 0.23795253783464432:
                                                                return 'W'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 0.003989797085523605:
                                                                if Q.max_dr > 0.18738361448049545:
                                                                    if Q.centroid_offset > 0.015454249456524849:
                                                                        if Q.z_dr_0p2_0p4 > 0.04483302682638168:
                                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 54% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 36.49642753601074:
                                                            if Q.sum_pt > 964.4375:
                                                                if Q.pt_7 > 40.375:
                                                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 26.5703125:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 38% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 6.480819138232619e-05:
                                                        if Q.LHA > 0.21589572727680206:
                                                            if Q.pt_7 > 35.265625:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_over_sum_pt > 0.048584477975964546:
                                                                return 'g'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 27.7109375:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 34.76131248474121:
                                            if Q.log_sum_pt > 6.9680562019348145:
                                                if Q.pt_7 > 21.7890625:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 40% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.0021254868479445577:
                                                    if Q.mass > 36.60154914855957:
                                                        if Q.pt_7 > 45.265625:
                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.0023870711447671056:
                                                                if Q.mass > 40.38593864440918:
                                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.LHA > 0.17040829360485077:
                                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.pt_7 > 28.859375:
                                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 35% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.08401632308959961:
                                                            return 'g'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.010215382557362318:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 34.375:
                                                            return 'g'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.eccentricity > 0.9274376034736633:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 619.21875:
                                                if Q.centroid_offset > 0.013146816287189722:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 26.7421875:
                                                        return 'g'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.012423703446984291:
                                    if Q.sum_pt > 1072.1796875:
                                        return 'g'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.013584989588707685:
                                            return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.573099821805954:
                                                if Q.tau21 > 0.3344530016183853:
                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.955458641052246:
                                        if Q.pt_7 > 15.94140625:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.004871866665780544:
                                                if Q.max_dr > 0.2655269354581833:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.009966480080038309:
                                            if Q.mass > 35.648067474365234:
                                                return 'W'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00011388678103685379:
                                                    return 'W'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 97% of the training jets here get this class from the formula
                else:
                    if Q.centroid_offset > 0.02600828278809786:
                        if Q.width > 0.004224083386361599:
                            if Q.max_dr > 0.11861122772097588:
                                if Q.mass > 37.68547439575195:
                                    if Q.pt_7 > 29.8046875:
                                        if Q.z_dr_0p1_0p2 > 0.22794073075056076:
                                            if Q.LHA > 0.2685113251209259:
                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 31% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.004747232189401984:
                                                if Q.centroid_offset > 0.04189164936542511:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.06350258737802505:
                                                        if Q.max_dr > 0.1216297410428524:
                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.2916877716779709:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.0245339535176754:
                                                    if Q.lam2 > 0.00036863771674688905:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.0046080732718110085:
                                                            if Q.e2 > 0.026109966449439526:
                                                                return 'W'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.11903106048703194:
                                                        if Q.width > 0.004438113188371062:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 675.1875:
                                            if Q.centroid_offset > 0.03815937042236328:
                                                return 't'   # 50% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.12997794151306152:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.029073969461023808:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 24.90625:
                                                if Q.lam1 > 0.005751354619860649:
                                                    if Q.e2 > 0.030125143937766552:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.0049814218655228615:
                                                        if Q.z_dr_0p1_0p2 > 0.15864123404026031:
                                                            return 't'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 47% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.width > 0.005462022498250008:
                                        if Q.tau21 > 0.2869793623685837:
                                            if Q.z_dr_0p05_0p1 > 0.3480300158262253:
                                                if Q.pt_7 > 31.8828125:
                                                    if Q.width > 0.005982909584417939:
                                                        if Q.mass > 33.92033576965332:
                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.03827075473964214:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 33.21875:
                                                    if Q.D2 > 0.8908135890960693:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.029579438269138336:
                                            if Q.pt_7 > 27.3828125:
                                                if Q.e2 > 0.02877629455178976:
                                                    if Q.z_dr_0p05_0p1 > 0.6509005427360535:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 31.234375:
                                                        if Q.e2 > 0.02528917882591486:
                                                            if Q.width > 0.004739392315968871:
                                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.0045790143776685:
                                                            return 't'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.2584008276462555:
                                                    return 'Z'   # 36% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.22910425812005997:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 33.375:
                                                    if Q.width > 0.004739655647426844:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.034901490435004234:
                                    if Q.sum_pt > 568.078125:
                                        if Q.centroid_offset > 0.04779483564198017:
                                            if Q.sum_pt > 668.4296875:
                                                return 'g'   # 38% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.2931225597858429:
                                                if Q.pt_7 > 27.34375:
                                                    if Q.tau21 > 0.09906633570790291:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.466749668121338:
                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.039459310472011566:
                                                                return 't'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.023685396648943424:
                                                    if Q.C2 > 0.020045487210154533:
                                                        if Q.max_dr > 0.10569922253489494:
                                                            return 'Z'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 8.781729775364511e-05:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.23168925195932388:
                                            if Q.LHA > 0.2917618602514267:
                                                if Q.log_sum_pt > 6.2034783363342285:
                                                    if Q.pt_7 > 35.890625:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.005455876234918833:
                                                if Q.sum_pt_top5 > 358.859375:
                                                    if Q.width > 0.006223120028153062:
                                                        return 't'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 35.50573539733887:
                                                            return 'Z'   # 47% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 45% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.005090527702122927:
                                                    return 'Z'   # 49% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 55% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.005599717143923044:
                                        if Q.mass > 43.71719932556152:
                                            if Q.max_dr > 0.10045789927244186:
                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.07875638827681541:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9818989932537079:
                                                        if Q.tau21 > 0.07571632787585258:
                                                            return 'W'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 0.5658393502235413:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 33.84375:
                                                if Q.e2 > 0.033712347969412804:
                                                    if Q.LHA > 0.31152884662151337:
                                                        if Q.pt_7 > 42.75:
                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 0.006410987349227071:
                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p1_0p2 > 0.2930692285299301:
                                                            return 'g'   # 44% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.0004268206830602139:
                                                                if Q.centroid_offset > 0.02979974076151848:
                                                                    return 'Z'   # 66% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth > 0.07629011571407318:
                                                                    if Q.e2 > 0.03606443293392658:
                                                                        return 'W'   # 78% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 513.15625:
                                                    if Q.max_dr > 0.1072445884346962:
                                                        return 't'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.06816093996167183:
                                                        return 'W'   # 35% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.10438399389386177:
                                            if Q.max_dr > 0.10733000189065933:
                                                if Q.tau21 > 0.17174769937992096:
                                                    if Q.z_dr_0p05_0p1 > 0.8122716248035431:
                                                        if Q.z_dr_0p1_0p2 > 0.09138427302241325:
                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.982915461063385:
                                                        if Q.centroid_offset > 0.029605559073388577:
                                                            if Q.width > 0.004841593327000737:
                                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.028135575354099274:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.030402916483581066:
                                                    if Q.tau21 > 0.16061103343963623:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.00012329139281064272:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 9.399819100508466e-05:
                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 41.4267463684082:
                                                    if Q.centroid_offset > 0.028254477307200432:
                                                        if Q.max_dr > 0.10406415164470673:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.03055714350193739:
                                                                return 'Z'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.03351897746324539:
                                if Q.max_dr > 0.13697384297847748:
                                    if Q.lam1 > 0.0032835203455761075:
                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.146794393658638:
                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.037916941568255424:
                                        if Q.eccentricity > 0.9630424976348877:
                                            if Q.e2 > 0.01675530057400465:
                                                if Q.mass > 32.28830909729004:
                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.8785712718963623:
                                            if Q.max_dr > 0.12011716887354851:
                                                if Q.girth2 > 0.0038458749186247587:
                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 65% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.14336436241865158:
                                    if Q.width > 0.0037435238482430577:
                                        if Q.centroid_offset > 0.027637316845357418:
                                            if Q.e2 > 0.021925599314272404:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.030663635581731796:
                                            if Q.mass > 34.78627395629883:
                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 933.484375:
                                        return 'g'   # 48% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.07705724611878395:
                                            if Q.centroid_offset > 0.03173012658953667:
                                                if Q.tau21 > 0.15314831584692:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00011642496247077361:
                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.1164914071559906:
                                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.027890090830624104:
                                                return 'Z'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                    else:
                        if Q.width > 0.0025453109992668033:
                            if Q.max_dr > 0.12358416616916656:
                                if Q.z_dr_0p05_0p1 > 0.565461128950119:
                                    if Q.centroid_offset > 0.012362429406493902:
                                        if Q.lam1 > 0.0054953983053565025:
                                            if Q.centroid_offset > 0.015094291418790817:
                                                if Q.mass > 44.81705093383789:
                                                    if Q.max_dr > 0.13017018139362335:
                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.2137356847524643:
                                                            return 'W'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.01656238827854395:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 55.460947036743164:
                                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 32.828125:
                                                        if Q.centroid_offset > 0.01863007340580225:
                                                            if Q.tau21 > 0.22661101818084717:
                                                                if Q.girth2 > 0.0061521276365965605:
                                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.13778413087129593:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 528.1796875:
                                                            return 'Z'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.0058538836892694235:
                                                    if Q.mass > 48.430179595947266:
                                                        if Q.max_dr > 0.12920185178518295:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 58.53921890258789:
                                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.1347956508398056:
                                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.1361292377114296:
                                                        if Q.sum_pt > 701.953125:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.020006952807307243:
                                                if Q.lam1 > 0.004764249548316002:
                                                    if Q.e2 > 0.02957253623753786:
                                                        if Q.girth2 > 0.005243360064923763:
                                                            if Q.mass > 44.29191780090332:
                                                                if Q.e2 > 0.03192237578332424:
                                                                    if Q.girth2 > 0.005624765995889902:
                                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.width > 0.005777437938377261:
                                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.13353395462036133:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam1 > 0.005036730784922838:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.027702393010258675:
                                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.005592373898252845:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.13621217757463455:
                                                            if Q.lam1 > 0.004574285354465246:
                                                                if Q.max_dr > 0.1408664733171463:
                                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.1305077224969864:
                                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.006092884577810764:
                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.005097057670354843:
                                                        if Q.e2 > 0.03028012253344059:
                                                            if Q.max_dr > 0.14887333661317825:
                                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.016097000800073147:
                                                                    if Q.sum_pt > 751.453125:
                                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.005767708411440253:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.028697915375232697:
                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.width > 0.004911964759230614:
                                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.006269819103181362:
                                            if Q.e2 > 0.03727709501981735:
                                                if Q.width > 0.006548668025061488:
                                                    if Q.log_sum_pt > 6.526831865310669:
                                                        if Q.max_dr > 0.12866134196519852:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.06450265273451805:
                                                                return 'W'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.13477367162704468:
                                                            return 'Z'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 68.90396118164062:
                                                        if Q.pt_7 > 25.359375:
                                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.14489348977804184:
                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.011093188542872667:
                                                                return 'W'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.log_sum_pt > 6.668178558349609:
                                                                    if Q.centroid_offset > 0.006981569807976484:
                                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.13284236937761307:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.008270729333162308:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 72.51177215576172:
                                                if Q.pt_7 > 24.1484375:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.006020301021635532:
                                                    if Q.e2 > 0.03461095690727234:
                                                        if Q.e2 > 0.03625890985131264:
                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.019787052646279335:
                                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.962324857711792:
                                        if Q.pt_7 > 48.71875:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 18.7265625:
                                                if Q.tau21 > 0.13714595884084702:
                                                    if Q.log_sum_pt > 7.029239654541016:
                                                        return 'g'   # 40% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p05_0p1 > 0.2889798581600189:
                                            if Q.max_dr > 0.14377819001674652:
                                                if Q.tau21 > 0.199649840593338:
                                                    if Q.LHA > 0.27542391419410706:
                                                        if Q.centroid_offset > 0.01389271067455411:
                                                            if Q.max_dr > 0.14646824449300766:
                                                                if Q.width > 0.00574914738535881:
                                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 51% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth > 0.06880009919404984:
                                                                if Q.max_dr > 0.1492953896522522:
                                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.023638421669602394:
                                                            if Q.width > 0.004469143692404032:
                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.009094411972910166:
                                                        if Q.max_dr > 0.14944154024124146:
                                                            if Q.tau21 > 0.134637713432312:
                                                                if Q.centroid_offset > 0.01618907693773508:
                                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.girth > 0.06622443720698357:
                                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p05_0p1 > 0.3650783598423004:
                                                                if Q.LHA > 0.27814699709415436:
                                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.021325942128896713:
                                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.017239520326256752:
                                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.00648064399138093:
                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.13670002669095993:
                                                    if Q.girth > 0.07047064229846:
                                                        if Q.centroid_offset > 0.005537985358387232:
                                                            if Q.log_sum_pt > 6.487111330032349:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.17479155212640762:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.017319314181804657:
                                                                if Q.z_dr_0p05_0p1 > 0.3872639387845993:
                                                                    if Q.mass > 50.48325729370117:
                                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.02141850534826517:
                                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth > 0.07486844435334206:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.856346845626831:
                                                            return 'W'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.022268262691795826:
                                                                if Q.lam1 > 0.004461977398023009:
                                                                    if Q.max_dr > 0.13255459815263748:
                                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 33.478933334350586:
                                                if Q.max_dr > 0.1519658863544464:
                                                    if Q.LHA > 0.26902058720588684:
                                                        if Q.centroid_offset > 0.014034813735634089:
                                                            if Q.z_dr_0p05_0p1 > 0.033978814259171486:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.27991625666618347:
                                                                if Q.z_dr_0p1_0p2 > 0.21393589675426483:
                                                                    if Q.width > 0.006399928592145443:
                                                                        return 'Z'   # 54% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.010312824044376612:
                                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.lam1 > 0.006315925158560276:
                                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 72.65265655517578:
                                                                    if Q.pt_7 > 23.984375:
                                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.009316114708781242:
                                                                        if Q.width > 0.006029469193890691:
                                                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.020685710944235325:
                                                            if Q.girth2 > 0.004353097639977932:
                                                                if Q.eccentricity > 0.9834658205509186:
                                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 46% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 1008.0234375:
                                                                return 'W'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 0.0011760437046177685:
                                                                    return 'W'   # 55% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.girth > 0.03346114605665207:
                                                                        if Q.z_dr_0p1_0p2 > 0.2436431124806404:
                                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.centroid_offset > 0.01821358036249876:
                                                                                if Q.lam1 > 0.004813361680135131:
                                                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 76.95255279541016:
                                                        if Q.pt_7 > 23.6484375:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0019266419694758952:
                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth > 0.03841824270784855:
                                                                if Q.lam2 > 0.00027461699210107327:
                                                                    if Q.D2 > 0.6260803043842316:
                                                                        if Q.girth2 > 0.006181117380037904:
                                                                            if Q.centroid_offset > 0.011411973740905523:
                                                                                if Q.D2 > 1.2152835726737976:
                                                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 70% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 96% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.centroid_offset > 0.023839052766561508:
                                                                                if Q.mass_over_sum_pt > 0.06041533872485161:
                                                                                    return 'Z'   # 44% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 49% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt > 1000.765625:
                                                                        if Q.z_7 > 0.042510123923420906:
                                                                            return 'Z'   # 40% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 93% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.z_dr_0p05_0p1 > 0.18324360251426697:
                                                                            if Q.max_dr > 0.14665915817022324:
                                                                                if Q.tau21 > 0.13264333456754684:
                                                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                                                else:
                                                                                    if Q.centroid_offset > 0.013344778213649988:
                                                                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.sum_pt > 928.9375:
                                                                                if Q.z_7 > 0.052355118095874786:
                                                                                    if Q.centroid_offset > 0.009588193148374557:
                                                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 35.66817092895508:
                                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.21394161134958267:
                                                    if Q.pt_7 > 28.4375:
                                                        if Q.z_dr_0p1_0p2 > 0.19828687608242035:
                                                            if Q.pt_7 > 34.796875:
                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.00945674255490303:
                                                                return 'W'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass_over_sum_pt > 0.06068536266684532:
                                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 578.0:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.013704539742320776:
                                                        return 'W'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                            else:
                                if Q.girth2 > 0.0031823458848521113:
                                    if Q.sum_pt > 1024.953125:
                                        if Q.pt_7 > 52.0:
                                            if Q.width > 0.003955870401114225:
                                                if Q.pt_7 > 59.609375:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 74.1427001953125:
                                                if Q.pt_7 > 25.1875:
                                                    if Q.sum_pt > 1258.59375:
                                                        return 't'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 79.0236930847168:
                                                        if Q.max_dr > 0.0979638583958149:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 7.037405014038086:
                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.01026088371872902:
                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.011656665243208408:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 1074.390625:
                                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.006484538083896041:
                                            if Q.e2 > 0.03915804624557495:
                                                if Q.centroid_offset > 0.014375132042914629:
                                                    if Q.log_sum_pt > 6.568866968154907:
                                                        if Q.width > 0.006608914118260145:
                                                            return 'Z'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.0982610397040844:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.006597865838557482:
                                                            if Q.e2 > 0.040081627666950226:
                                                                if Q.lam1 > 0.00619017519056797:
                                                                    if Q.mass > 51.2689151763916:
                                                                        return 'Z'   # 61% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.020860079675912857:
                                                                if Q.lam2 > 0.0002880152896977961:
                                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 75.89169311523438:
                                                        if Q.pt_7 > 27.4140625:
                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0014675973798148334:
                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.03978605568408966:
                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.007472691358998418:
                                                                    if Q.mass > 58.93243980407715:
                                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.013096474576741457:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.0065431196708232164:
                                                        if Q.log_sum_pt > 6.629195928573608:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 33.24942970275879:
                                                if Q.centroid_offset > 0.018720555119216442:
                                                    if Q.z_dr_0p1_0p2 > 0.20439542829990387:
                                                        if Q.LHA > 0.2953052967786789:
                                                            if Q.e2 > 0.037598658353090286:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.005867386003956199:
                                                                    if Q.sum_pt_top5 > 487.375:
                                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.lam1 > 0.006140500772744417:
                                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 41% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.max_dr > 0.11449002847075462:
                                                                        if Q.e2 > 0.032849567010998726:
                                                                            return 'W'   # 49% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.11908488720655441:
                                                                if Q.z_dr_0p05_0p1 > 0.7253382503986359:
                                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 58.48423767089844:
                                                            if Q.z_dr_0p1_0p2 > 0.054450567811727524:
                                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.02083807159215212:
                                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.006079232785850763:
                                                                if Q.e2 > 0.036836059764027596:
                                                                    if Q.planar_flow > 0.18091589212417603:
                                                                        if Q.tau21 > 0.1400471329689026:
                                                                            if Q.planar_flow > 0.5685557425022125:
                                                                                return 'Z'   # 48% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 96% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.max_dr > 0.10145179554820061:
                                                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 52.69331359863281:
                                                                    if Q.centroid_offset > 0.023813113570213318:
                                                                        if Q.tau21 > 0.08480548113584518:
                                                                            return 'W'   # 65% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0p1_0p2 > 0.18417925387620926:
                                                                        if Q.centroid_offset > 0.02445139456540346:
                                                                            if Q.max_dr > 0.1158440075814724:
                                                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 87% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 77.30244445800781:
                                                        if Q.pt_7 > 31.390625:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.0035733843687921762:
                                                            if Q.centroid_offset > 0.014670164790004492:
                                                                if Q.mass > 63.23426628112793:
                                                                    if Q.max_dr > 0.11497028172016144:
                                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.lam1 > 0.00610301923006773:
                                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam1 > 0.006232420448213816:
                                                                        if Q.e2 > 0.037265678867697716:
                                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.mass > 58.80743408203125:
                                                                            if Q.z_dr_0p05_0p1 > 0.7902135848999023:
                                                                                return 'W'   # 99% of the training jets here get this class from the formula
                                                                            else:
                                                                                if Q.centroid_offset > 0.01695371512323618:
                                                                                    return 'Z'   # 61% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.girth2 > 0.005948502337560058:
                                                                                if Q.e2 > 0.03525761887431145:
                                                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 78.21875:
                                                                    return 'W'   # 57% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 73.78594207763672:
                                                                        if Q.centroid_offset > 0.0068617628421634436:
                                                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 60.265625:
                                                                if Q.log_sum_pt > 6.782156467437744:
                                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top5 > 472.734375:
                                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.007074704859405756:
                                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.01118038222193718:
                                                    if Q.z_dr_0p1_0p2 > 0.2367071881890297:
                                                        if Q.pt_7 > 32.671875:
                                                            if Q.z_dr_0p05_0p1 > 0.14290476590394974:
                                                                return 'W'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 28.28125:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 493.0859375:
                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.0038305744528770447:
                                                        if Q.z_7 > 0.06747649982571602:
                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 29.5234375:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 31.63897132873535:
                                                            if Q.width > 0.003581195021979511:
                                                                if Q.C2 > 0.023960988968610764:
                                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.008988149464130402:
                                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.011155994608998299:
                                        if Q.sum_pt_top5 > 463.984375:
                                            if Q.pt_7 > 56.453125:
                                                if Q.log_sum_pt > 6.786998748779297:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.3418489098548889:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 53.976640701293945:
                                                    return 'g'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 645.3359375:
                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.014203758910298347:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 0.0028136305045336485:
                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.016361559741199017:
                                                return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.002857682411558926:
                                                    if Q.sum_pt_top5 > 442.875:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.049819398671388626:
                                            if Q.mass > 36.724937438964844:
                                                if Q.pt_7 > 51.765625:
                                                    if Q.log_sum_pt > 6.740741014480591:
                                                        return 'g'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 0.6668290197849274:
                                                            return 'W'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.0028483063215389848:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.007189703173935413:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.008980788290500641:
                                                    if Q.mass > 32.738142013549805:
                                                        if Q.z_7 > 0.06587276980280876:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 41.40625:
                                                if Q.mass > 66.13613891601562:
                                                    return 't'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 39.183311462402344:
                                                    if Q.mass > 65.19822692871094:
                                                        return 't'   # 43% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.005274764960631728:
                                                        if Q.mass > 34.58474349975586:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 39% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.012802626937627792:
                                if Q.pt_7 > 47.921875:
                                    if Q.log_sum_pt > 6.812809228897095:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.9884721636772156:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.017217795364558697:
                                                if Q.pt_7 > 53.8125:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.944222927093506:
                                        if Q.pt_7 > 32.71875:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.1936148703098297:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.014584976714104414:
                                            if Q.sum_pt > 968.1875:
                                                if Q.z_7 > 0.03705652058124542:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 31.858102798461914:
                                                if Q.e2 > 0.01261582924053073:
                                                    if Q.pt_7 > 39.640625:
                                                        if Q.mass_over_sum_pt > 0.04588671028614044:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 33.484375:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00014882882533129305:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 75% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 35.796875:
                                    if Q.pt_7 > 43.671875:
                                        if Q.eccentricity > 0.9911127984523773:
                                            if Q.C2 > 0.016940240748226643:
                                                if Q.log_sum_pt > 6.777575492858887:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.003646953613497317:
                                                return 'g'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 31.7706937789917:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 0.8301079273223877:
                                                        return 'g'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.007766424911096692:
                                            if Q.mass > 34.17051696777344:
                                                if Q.sum_pt > 987.828125:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 37.02006149291992:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.0877145566046238:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 2.4412469429080375e-05:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.0018169907270930707:
                                                        return 'g'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 991.8125:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.0017324390937574208:
                                                    if Q.centroid_offset > 0.004064811626449227:
                                                        if Q.z_7 > 0.05261795036494732:
                                                            return 'g'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 36.616472244262695:
                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 38.859375:
                                                            if Q.e2 > 0.024761293083429337:
                                                                return 'g'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.LHA > 0.2081456407904625:
                                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.964836835861206:
                                        if Q.pt_7 > 19.578125:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.0020172555232420564:
                                            if Q.centroid_offset > 0.007295712595805526:
                                                if Q.mass > 35.72781944274902:
                                                    if Q.mass > 38.08780479431152:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.03371983952820301:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.04629606939852238:
                                                        return 'g'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.03487794101238251:
                                                            return 'g'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.011550191324204206:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.03123100008815527:
                                                    if Q.mass > 35.295772552490234:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.0024698490742594004:
                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 829.1875:
                                                            if Q.girth2 > 0.002292528748512268:
                                                                if Q.centroid_offset > 0.004388092085719109:
                                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.010707107838243246:
                                                if Q.mass > 36.70188331604004:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00018949544755741954:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 97% of the training jets here get this class from the formula
        else:
            if Q.sum_pt_top5 > 600.140625:
                if Q.centroid_offset > 0.015202948357909918:
                    if Q.centroid_offset > 0.025169466622173786:
                        if Q.mass_over_sum_pt > 0.02235360909253359:
                            if Q.centroid_offset > 0.03420642949640751:
                                if Q.pt_7 > 24.390625:
                                    if Q.centroid_offset > 0.053833985701203346:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.09403764456510544:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.03789863362908363:
                                                return 'Z'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.595321416854858:
                                        if Q.e2 > 0.009033254813402891:
                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 50% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 59% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 50.84375:
                                    return 'g'   # 77% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.030341903679072857:
                                        if Q.LHA > 0.2240791618824005:
                                            return 'W'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.00024305967963300645:
                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.664329290390015:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.75286340713501:
                                            if Q.centroid_offset > 0.027417962439358234:
                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 92% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.041205454617738724:
                                if Q.pt_7 > 21.640625:
                                    if Q.pt_7 > 46.8125:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.05549854598939419:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_0 > 364.125:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p05_0p1 > 0.5982211530208588:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 744.125:
                                                        if Q.pt_7 > 37.46875:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.04333779215812683:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 45% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 1007.140625:
                                    return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.635361909866333:
                                        if Q.pt_7 > 56.53125:
                                            if Q.mass > 5.651435852050781:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.026713263243436813:
                                                if Q.mass_over_sum_pt > 0.013289661146700382:
                                                    if Q.centroid_offset > 0.028480405919253826:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.6823790073394775:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.21808244287967682:
                                                        if Q.pt_0 > 347.75:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 789.421875:
                                                    if Q.lam1 > 0.0007973083993420005:
                                                        if Q.centroid_offset > 0.025798614136874676:
                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 807.796875:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 32.21875:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.003333748783916235:
                                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.04223934933543205:
                                                        if Q.mass_over_sum_pt > 0.00969327287748456:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.028484536334872246:
                                            if Q.pt_7 > 26.4609375:
                                                if Q.log_sum_pt > 6.5849316120147705:
                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.029990903101861477:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 22.6171875:
                                                    if Q.centroid_offset > 0.030566648580133915:
                                                        return 'Z'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.595599174499512:
                                                        return 'Z'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 23.53125:
                                                if Q.pt_7 > 34.421875:
                                                    if Q.centroid_offset > 0.025956567376852036:
                                                        if Q.mass_over_sum_pt > 0.010969303548336029:
                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.027257222682237625:
                                                        if Q.pt_7 > 29.75:
                                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.597103118896484:
                                                    return 'W'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 41.078125:
                            if Q.centroid_offset > 0.01942965481430292:
                                if Q.pt_7 > 51.328125:
                                    if Q.mass > 5.010485649108887:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.023440031334757805:
                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.792465925216675:
                                                return 'g'   # 54% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.07147317379713058:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.761482238769531:
                                        if Q.sum_pt > 1017.671875:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.021670828573405743:
                                                if Q.e2 > 0.005340077448636293:
                                                    return 'W'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 763.90625:
                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.023922153748571873:
                                            if Q.sum_pt > 817.421875:
                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 614.078125:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.021208468824625015:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.z_7 > 0.05407501943409443:
                                    if Q.centroid_offset > 0.018119903281331062:
                                        if Q.z_7 > 0.05711294151842594:
                                            if Q.sum_pt_top5 > 657.5:
                                                if Q.mass > 4.648812770843506:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.029464857652783394:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.017524785362184048:
                                        if Q.sum_pt > 1009.703125:
                                            return 'g'   # 100% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 984.84375:
                                            return 'g'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.05053334683179855:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.016345469281077385:
                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.017510971054434776:
                                if Q.log_sum_pt > 6.613069534301758:
                                    if Q.centroid_offset > 0.023481263779103756:
                                        if Q.log_sum_pt > 6.737207889556885:
                                            if Q.girth > 0.027524849399924278:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 862.3359375:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 765.15625:
                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.024385194294154644:
                                                            return 'Z'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2_sq > 3.156591810693499e-05:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 812.140625:
                                                if Q.centroid_offset > 0.024292937479913235:
                                                    if Q.lam1 > 0.0006707741995342076:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 28.2109375:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.964061737060547:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.655711889266968:
                                                if Q.sum_pt > 937.7421875:
                                                    if Q.eccentricity > 0.9697472155094147:
                                                        if Q.centroid_offset > 0.020688090473413467:
                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 28.4296875:
                                                                return 'Z'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.eccentricity > 0.9833643734455109:
                                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.02174477931112051:
                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.022477091290056705:
                                                        if Q.log_sum_pt > 6.790633201599121:
                                                            if Q.lam1 > 0.0005727002571802586:
                                                                return 'W'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 786.82421875:
                                                            if Q.planar_flow > 0.08691059798002243:
                                                                return 'W'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.log_sum_pt > 6.819487571716309:
                                                                    if Q.centroid_offset > 0.021258985623717308:
                                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 23.546875:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.019738780334591866:
                                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.15788796544075012:
                                                    if Q.pt_7 > 23.7421875:
                                                        if Q.centroid_offset > 0.01979957055300474:
                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 618.59375:
                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.1915702074766159:
                                                            return 'W'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.020885786972939968:
                                                                return 'W'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.03770332597196102:
                                                        if Q.sum_pt > 766.796875:
                                                            return 'W'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.03313957713544369:
                                        if Q.LHA > 0.17167958617210388:
                                            if Q.sum_pt > 709.4921875:
                                                if Q.centroid_offset > 0.02022167295217514:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth > 0.027774933725595474:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.19225046038627625:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.020936849527060986:
                                                if Q.sum_pt > 733.5390625:
                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.02188266161829233:
                                                        return 'W'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.03838410973548889:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 2.5854786144918762e-05:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 58% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.19836344569921494:
                                            if Q.log_sum_pt > 6.576817035675049:
                                                return 'W'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 811.0703125:
                                    if Q.sum_pt > 1063.32421875:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.01597284898161888:
                                            if Q.eccentricity > 0.9279937446117401:
                                                if Q.log_sum_pt > 6.76875376701355:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 28.578125:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.016944597475230694:
                                                            if Q.pt_7 > 23.28125:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.2638038545846939:
                                                if Q.pt_7 > 23.6015625:
                                                    if Q.pt_7 > 37.203125:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.781224489212036:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.5833396911621094:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 34.46875:
                                                    if Q.z_7 > 0.04300428368151188:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 876.8125:
                                                        if Q.pt_7 > 25.0234375:
                                                            return 'W'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 944.171875:
                                                                return 'W'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.042376887053251266:
                                        if Q.mass > 25.574748039245605:
                                            return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.04382506385445595:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 32.765625:
                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.4481119215488434:
                                            if Q.pt_7 > 25.7421875:
                                                if Q.sum_pt > 774.078125:
                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.17065750807523727:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 21.708215713500977:
                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 27.2248477935791:
                                                if Q.pt_7 > 26.046875:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.03755089081823826:
                                                    if Q.sum_pt_top5 > 638.375:
                                                        if Q.centroid_offset > 0.01631895825266838:
                                                            if Q.log_sum_pt > 6.639600038528442:
                                                                return 'W'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.15673521161079407:
                                                            return 'q'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.0003123758651781827:
                                                                return 'g'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 95% of the training jets here get this class from the formula
                else:
                    if Q.pt_7 > 43.671875:
                        if Q.centroid_offset > 0.0054222275502979755:
                            if Q.width > 9.797033999348059e-05:
                                if Q.pt_7 > 47.796875:
                                    if Q.centroid_offset > 0.0069850285071879625:
                                        return 'g'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 49.78125:
                                            return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.00897508766502142:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 2.007083730859449e-05:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.00013295959797687829:
                                        if Q.centroid_offset > 0.007831163238734007:
                                            return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.1472190618515015:
                                                if Q.lam2 > 3.9650985854677856e-05:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.057092269882559776:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.848767518997192:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.01482699578627944:
                                                                return 'g'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0059091083239763975:
                                                    if Q.tau21 > 0.20959309488534927:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 649.6875:
                                            if Q.sum_pt_top5 > 806.75:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 52.140625:
                                    if Q.pt_7 > 54.640625:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 6.975772703299299e-05:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 64% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1004.9375:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.06068715080618858:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.654595375061035:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.913753032684326:
                                if Q.sum_pt > 1021.78125:
                                    return 'g'   # 100% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 51.078125:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 1.2049371434841305e-05:
                                            return 'g'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 93% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 60.640625:
                                    if Q.centroid_offset > 0.002554232720285654:
                                        if Q.girth2 > 4.657395402318798e-05:
                                            if Q.lam2 > 1.3464138646668289e-05:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 63.640625:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.006652002921327949:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 65.75:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.008640520740300417:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 70.46875:
                                                if Q.lam1 > 1.4264119272411335e-05:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 975.15625:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.011944939848035574:
                                        if Q.pt_7 > 50.890625:
                                            if Q.centroid_offset > 0.0014100890839472413:
                                                if Q.C2 > 0.013787661213427782:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.0029881199588999152:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 54% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.01721278578042984:
                                                if Q.pt_7 > 44.828125:
                                                    if Q.centroid_offset > 0.0037255478091537952:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.45779629051685333:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 3.615052446548361e-05:
                                                    if Q.girth > 0.01950717903673649:
                                                        return 'q'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 653.171875:
                                                            return 'q'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.00397106003947556:
                                            if Q.z_7 > 0.06210026703774929:
                                                if Q.C2 > 0.005006321240216494:
                                                    if Q.lam2 > 2.7749471882998478e-05:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 54.984375:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_7 > 0.065322145819664:
                                                                return 'g'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.006900922395288944:
                                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 52.296875:
                                                    if Q.C2 > 0.006016850005835295:
                                                        if Q.pt_7 > 54.171875:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 2.4886795472411904e-05:
                                                if Q.pt_7 > 54.78125:
                                                    if Q.centroid_offset > 0.002632948337122798:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 99% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 1090.4140625:
                            if Q.pt_7 > 21.7890625:
                                if Q.pt_7 > 27.5703125:
                                    if Q.log_sum_pt > 7.003320932388306:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 30.6328125:
                                            return 'g'   # 100% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 54% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1125.578125:
                                        if Q.centroid_offset > 0.002459490788169205:
                                            return 'g'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.007200142135843635:
                                                return 'q'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.0036458132090047:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.004916489589959383:
                                    if Q.pt_7 > 16.1171875:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 19.4296875:
                                        if Q.centroid_offset > 0.003557738964445889:
                                            return 'g'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 20.5078125:
                                                if Q.sum_pt > 1132.8359375:
                                                    if Q.sum_pt_top5 > 1122.09375:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 100% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 99% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 100% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.009808523580431938:
                                if Q.z_7 > 0.04390028305351734:
                                    if Q.width > 0.00018179945618612692:
                                        if Q.z_7 > 0.04618147574365139:
                                            if Q.tau21 > 0.3158039003610611:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 3.5294191548018716e-05:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.011406481731683016:
                                                        return 'g'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.8821099102497101:
                                                if Q.centroid_offset > 0.013418358750641346:
                                                    if Q.girth > 0.019100790843367577:
                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 865.734375:
                                                        return 'g'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.04993216507136822:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.629322052001953:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 66% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 37.546875:
                                        if Q.log_sum_pt > 6.869112014770508:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.00022398400324163958:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.944463729858398:
                                            if Q.pt_7 > 25.828125:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.013533343560993671:
                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 7.675119923078455e-05:
                                                if Q.z_7 > 0.037804070860147476:
                                                    if Q.LHA > 0.16290704905986786:
                                                        if Q.lam2 > 0.00017556043894728646:
                                                            return 'g'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.01207850594073534:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.631682395935059:
                                                                return 'q'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.013877674471586943:
                                                        if Q.log_sum_pt > 6.737837791442871:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.00021992361871525645:
                                                                return 'W'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top5 > 635.25:
                                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 626.1875:
                                                            return 'q'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.15645717084407806:
                                                                return 'q'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.014124136418104172:
                                                    if Q.planar_flow > 0.4569671005010605:
                                                        if Q.log_sum_pt > 6.712547779083252:
                                                            if Q.z_7 > 0.020927174016833305:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_7 > 0.03481018915772438:
                                                                return 'g'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.860630989074707:
                                                            if Q.z_7 > 0.02575570624321699:
                                                                return 'W'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.04003036953508854:
                                                        if Q.lam2 > 4.4999056626693346e-05:
                                                            if Q.LHA > 0.15211321413516998:
                                                                return 'q'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.011535325553268194:
                                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 35.78125:
                                                            return 'g'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top5 > 631.765625:
                                    if Q.sum_pt > 1051.716796875:
                                        if Q.pt_7 > 32.296875:
                                            if Q.pt_7 > 34.859375:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.00567845324985683:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.00653677829541266:
                                                if Q.pt_7 > 25.375:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 38.171875:
                                            if Q.log_sum_pt > 6.935096740722656:
                                                if Q.sum_pt > 1038.734375:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.0032459619687870145:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 5.107382276037242e-05:
                                                    if Q.centroid_offset > 0.0064793918281793594:
                                                        if Q.z_7 > 0.04519137926399708:
                                                            if Q.D2 > 2.064151644706726:
                                                                return 'g'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 6.928291986696422e-05:
                                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.008052298799157143:
                                                        if Q.lam2 > 2.7293519451632164e-05:
                                                            if Q.pt_7 > 41.921875:
                                                                return 'g'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_7 > 0.048561545088887215:
                                                                    return 'g'   # 53% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.01576661691069603:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.636571645736694:
                                                if Q.z_7 > 0.046862635761499405:
                                                    if Q.lam2 > 6.256287815631367e-05:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 34.203125:
                                                        if Q.log_sum_pt > 6.942050933837891:
                                                            if Q.girth > 0.006801859708502889:
                                                                return 'g'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 7.430182449752465e-05:
                                                                if Q.centroid_offset > 0.008895405102521181:
                                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 3.953142550017219e-05:
                                                    if Q.centroid_offset > 0.0014461571699939668:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 6.904740810394287:
                                                            return 'q'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 25.890625:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 39% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.0017684285412542522:
                                                        return 'q'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 744.578125:
                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 44% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.007039215415716171:
                                        if Q.z_7 > 0.04561944492161274:
                                            if Q.lam2 > 3.710939017764758e-05:
                                                if Q.tau21 > 0.2873910367488861:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.054545020684599876:
                                                    if Q.D2 > 1.4406113624572754:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt > 0.0058701834641397:
                                                        if Q.centroid_offset > 0.009213371202349663:
                                                            return 'g'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 2.3179194927215576:
                                                                return 'g'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.LHA > 0.12267598882317543:
                                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.007768378360196948:
                                                                        if Q.z_7 > 0.049988171085715294:
                                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 73% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 6.46901098662056e-05:
                                                if Q.girth > 0.014839510433375835:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.0020282540936022997:
                                            if Q.lam2 > 4.844830800720956e-05:
                                                if Q.z_7 > 0.0483438428491354:
                                                    if Q.D2 > 1.44184148311615:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 2.3556193113327026:
                                                        if Q.z_7 > 0.04138057865202427:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.04672048054635525:
                                                    return 'q'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.55434513092041:
                                                        return 'q'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.003851172514259815:
                                                            return 'q'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 9.711723032523878e-05:
                                                                return 'q'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 6.934298992156982:
                                                if Q.mass > 8.691399574279785:
                                                    return 'q'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 730.4375:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 44% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 759.609375:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.575870990753174:
                                                        if Q.planar_flow > 0.4486275464296341:
                                                            if Q.log_sum_pt > 6.592402219772339:
                                                                if Q.centroid_offset > 0.001427832234185189:
                                                                    if Q.log_sum_pt > 6.612576246261597:
                                                                        return 'q'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
            else:
                if Q.sum_pt_top5 > 541.015625:
                    if Q.centroid_offset > 0.02323187328875065:
                        if Q.centroid_offset > 0.029145861975848675:
                            if Q.centroid_offset > 0.040182966738939285:
                                if Q.mass_over_sum_pt > 0.01623780932277441:
                                    if Q.centroid_offset > 0.0531453900039196:
                                        if Q.e2_sq > 0.0010632798075675964:
                                            return 't'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 24.3984375:
                                            if Q.max_dr > 0.11192864179611206:
                                                if Q.pt_7 > 30.734375:
                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 48% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.004481861367821693:
                                        if Q.pt_7 > 35.390625:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.004578196443617344:
                                            if Q.max_dr > 0.06256085261702538:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.062039969488978386:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.044457847252488136:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p05_0p1 > 0.13053133338689804:
                                                if Q.girth2 > 0.003220300772227347:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.6940080225467682:
                                                        return 'Z'   # 49% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt > 0.02021301444619894:
                                    if Q.centroid_offset > 0.03626507148146629:
                                        if Q.eccentricity > 0.9843012392520905:
                                            if Q.mass > 25.771214485168457:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 46% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 23.453125:
                                            if Q.lam2 > 0.00026594894006848335:
                                                if Q.e2 > 0.015056186821311712:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.03180980309844017:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 46% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.03314349241554737:
                                                    if Q.girth > 0.04364968277513981:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 32.90625:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 21.390625:
                                                return 't'   # 37% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 27.4921875:
                                        if Q.centroid_offset > 0.03169715031981468:
                                            if Q.z_7 > 0.07062621042132378:
                                                if Q.girth2 > 0.0013234414509497583:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 34.25:
                                                if Q.lam1 > 0.001117685402277857:
                                                    return 'W'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 701.4375:
                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 23.0078125:
                                            if Q.sum_pt > 657.78125:
                                                if Q.lam2 > 5.1068489483441226e-05:
                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.03413373418152332:
                                                        return 'g'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 95% of the training jets here get this class from the formula
                        else:
                            if Q.LHA > 0.19447589665651321:
                                if Q.pt_7 > 24.046875:
                                    if Q.pt_7 > 51.609375:
                                        return 'g'   # 59% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_over_sum_pt > 0.01547615835443139:
                                            if Q.pt_7 > 26.8515625:
                                                return 'W'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.0013429682003334165:
                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 53% of the training jets here get this class from the formula
                                else:
                                    if Q.eccentricity > 0.8986242413520813:
                                        return 'q'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 52% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 686.65625:
                                    if Q.z_7 > 0.05889381095767021:
                                        if Q.centroid_offset > 0.02581479772925377:
                                            if Q.e2 > 0.005908712046220899:
                                                if Q.z_7 > 0.06557955965399742:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 726.796875:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.027146521024405956:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 569.40625:
                                                if Q.planar_flow > 0.22833364456892014:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.02703790832310915:
                                            if Q.pt_7 > 35.390625:
                                                if Q.sum_pt > 708.40625:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 700.5390625:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.1787158027291298:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.18613124638795853:
                                        if Q.pt_7 > 29.421875:
                                            if Q.sum_pt_top5 > 551.203125:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 42% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.026208246126770973:
                                            if Q.pt_7 > 31.59375:
                                                return 'W'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.005501128500327468:
                            if Q.z_7 > 0.04625384882092476:
                                if Q.LHA > 0.18996033817529678:
                                    if Q.centroid_offset > 0.01887026336044073:
                                        if Q.width > 0.0017524597351439297:
                                            return 'W'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.05672169290482998:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.020712236873805523:
                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 21.392415046691895:
                                                        return 'W'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.05392457917332649:
                                            if Q.centroid_offset > 0.009071070235222578:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.05856258422136307:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.012146254070103168:
                                                if Q.mass > 26.78056049346924:
                                                    if Q.centroid_offset > 0.015663775615394115:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.020911420695483685:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.00785514060407877:
                                        if Q.centroid_offset > 0.02241442445665598:
                                            if Q.sum_pt_top5 > 574.28125:
                                                if Q.z_7 > 0.05936853028833866:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.05048511177301407:
                                                if Q.centroid_offset > 0.019827033393085003:
                                                    if Q.sum_pt_top5 > 580.21875:
                                                        if Q.z_7 > 0.05727392062544823:
                                                            return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 727.875:
                                                                return 'W'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.009969784412533045:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.575903654098511:
                                                        if Q.eccentricity > 0.8922093212604523:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.1531396359205246:
                                                            return 'q'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.05691233649849892:
                                            if Q.tau21 > 0.25990137457847595:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.060515373945236206:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.39521007239818573:
                                                if Q.log_sum_pt > 6.587356328964233:
                                                    if Q.lam2 > 2.974213111883728e-05:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.5092415511608124:
                                                        return 'g'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 573.1875:
                                                            if Q.z_7 > 0.052272846922278404:
                                                                return 'g'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.15677059441804886:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt > 0.022538483142852783:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.05406473018229008:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 80% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.011139838490635157:
                                    if Q.LHA > 0.1920303851366043:
                                        if Q.centroid_offset > 0.01815138664096594:
                                            if Q.pt_7 > 25.3125:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00011233153782086447:
                                                    return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.0001968592987395823:
                                                return 'W'   # 38% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 1.7861970663070679:
                                                    return 'g'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.0314179640263319:
                                            if Q.lam2 > 3.8347954614437185e-05:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 10.3250093460083:
                                                    if Q.log_sum_pt > 6.497611045837402:
                                                        if Q.pt_7 > 29.8359375:
                                                            return 'g'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.567254543304443:
                                                        if Q.lam1 > 0.00028718971589114517:
                                                            return 'g'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.planar_flow > 0.45068077743053436:
                                        if Q.sum_pt > 686.109375:
                                            if Q.centroid_offset > 0.009293806739151478:
                                                if Q.pt_7 > 29.765625:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.012521013617515564:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.03152359277009964:
                                                if Q.sum_pt_top5 > 588.75:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.007084443932399154:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.4810144901275635:
                                                            if Q.z_7 > 0.04358902387320995:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p1_0p2 > 0.03075181320309639:
                                            if Q.planar_flow > 0.1655079573392868:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.1400144398212433:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 576.0:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.008372006472200155:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 72% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top5 > 576.109375:
                                if Q.z_7 > 0.07749704271554947:
                                    if Q.width > 3.044366076210281e-05:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 1.6109756827354431:
                                        if Q.z_7 > 0.05189843103289604:
                                            if Q.lam2 > 2.0162715372862294e-05:
                                                if Q.centroid_offset > 0.003922490868717432:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.01559755438938737:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.05861110799014568:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.10087266564369202:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.013416907750070095:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 583.8125:
                                                        if Q.centroid_offset > 0.0017873531323857605:
                                                            if Q.centroid_offset > 0.00397174246609211:
                                                                if Q.z_7 > 0.06107975170016289:
                                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.618258953094482:
                                                                return 'q'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 748.625:
                                                            return 'q'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.0023189398925751448:
                                                if Q.LHA > 0.08371403068304062:
                                                    if Q.mass > 15.101137638092041:
                                                        if Q.z_7 > 0.04382969066500664:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 705.5234375:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.4247862547636032:
                                                                if Q.LHA > 0.10155412182211876:
                                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt_top5 > 587.25:
                                                                        return 'q'   # 75% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 707.5:
                                                        return 'q'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 10.390383243560791:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 724.5:
                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 1.105665683746338:
                                            if Q.centroid_offset > 0.00448684929870069:
                                                if Q.z_7 > 0.06118369847536087:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.05086393840610981:
                                                        if Q.log_sum_pt > 6.594755411148071:
                                                            return 'q'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0017082500271499157:
                                                    if Q.z_7 > 0.06302154064178467:
                                                        if Q.lam2 > 1.8801952137437183e-05:
                                                            if Q.centroid_offset > 0.002996908617205918:
                                                                return 'g'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.021298721432685852:
                                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.09955013915896416:
                                                            return 'q'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.579576730728149:
                                                                return 'q'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.0029546577716246247:
                                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.609791278839111:
                                                        if Q.sum_pt > 764.328125:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.0060776446480304:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 7.803250312805176:
                                                            return 'q'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.061766959726810455:
                                                if Q.centroid_offset > 0.004214153159409761:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.tau21 > 0.4282694458961487:
                                    if Q.z_7 > 0.055964576080441475:
                                        if Q.sum_pt > 739.359375:
                                            if Q.z_7 > 0.07468939572572708:
                                                if Q.centroid_offset > 0.002612668205983937:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.08309522271156311:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 761.484375:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0028724631993100047:
                                                    if Q.z_7 > 0.06428424641489983:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.001764593762345612:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.6232969760894775:
                                                            return 'q'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.2342994809150696:
                                                if Q.tau21 > 0.5300849378108978:
                                                    if Q.z_7 > 0.05971803329885006:
                                                        return 'g'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 712.109375:
                                                            if Q.centroid_offset > 0.0039060034323483706:
                                                                return 'g'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.0022809868678450584:
                                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.010185576975345612:
                                                    if Q.centroid_offset > 0.003633708111010492:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 702.734375:
                                            if Q.centroid_offset > 0.0021809471072629094:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.581462144851685:
                                                    return 'W'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.10817525535821915:
                                                if Q.z_7 > 0.04795434512197971:
                                                    if Q.log_sum_pt > 6.513716697692871:
                                                        if Q.tau21 > 0.5311771631240845:
                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.6925275325775146:
                                                        if Q.sum_pt > 664.9765625:
                                                            return 'q'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.021182237192988396:
                                                                return 'q'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.003289955318905413:
                                                    if Q.z_7 > 0.04980484023690224:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 670.7734375:
                                                            return 'q'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.06393412873148918:
                                        if Q.centroid_offset > 0.0035950278397649527:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.0770084336400032:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 1.2866547107696533:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 0.9677151739597321:
                                                        if Q.sum_pt > 724.1875:
                                                            return 'q'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 1.1915335655212402:
                                            if Q.z_7 > 0.05759574845433235:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.08535563200712204:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.1424507424235344:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.04959381930530071:
                                                            return 'g'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 7.84915566444397:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 54% of the training jets here get this class from the formula
                else:
                    if Q.mass > 25.665185928344727:
                        if Q.centroid_offset > 0.01871349010616541:
                            if Q.log_sum_pt > 6.245288848876953:
                                if Q.centroid_offset > 0.035397542640566826:
                                    if Q.centroid_offset > 0.04749254882335663:
                                        if Q.girth > 0.06835401058197021:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 36.734375:
                                                return 'g'   # 38% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.12912877649068832:
                                            if Q.max_dr > 0.16640623658895493:
                                                return 't'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.033138152211904526:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 47% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 9.752220285008661e-05:
                                                if Q.tau21 > 0.2321499139070511:
                                                    if Q.centroid_offset > 0.04005540907382965:
                                                        return 'Z'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 27.80210018157959:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.043545763939619064:
                                                    return 'Z'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.08112458512187004:
                                                        return 'Z'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.21896075457334518:
                                        if Q.max_dr > 0.16142454743385315:
                                            if Q.log_sum_pt > 6.341295480728149:
                                                return 'W'   # 45% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.02111402153968811:
                                                if Q.log_sum_pt > 6.293390274047852:
                                                    if Q.pt_7 > 25.4765625:
                                                        if Q.centroid_offset > 0.03354540280997753:
                                                            if Q.lam2 > 0.00020618356938939542:
                                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 38% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.003185554058291018:
                                                        if Q.centroid_offset > 0.030560960993170738:
                                                            if Q.girth > 0.05697593465447426:
                                                                return 'W'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 27.84430980682373:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.06473644450306892:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 591.8046875:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 482.109375:
                                            if Q.LHA > 0.19888490438461304:
                                                if Q.z_dr_0p1_0p2 > 0.043907662853598595:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 32.578125:
                                    if Q.girth2 > 0.005382688250392675:
                                        if Q.e2 > 0.02422923594713211:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 27.265528678894043:
                                            if Q.z_dr_0p1_0p2 > 0.20760828256607056:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.031149443238973618:
                                                    if Q.max_dr > 0.1158955730497837:
                                                        if Q.pt_7 > 36.890625:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 27% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.14814170449972153:
                                                        return 'g'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p05_0p1 > 0.6946088373661041:
                                                if Q.max_dr > 0.09716220945119858:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 5.344550845620688e-05:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.025329873897135258:
                                        if Q.mass > 28.23387908935547:
                                            if Q.C2 > 0.045074064284563065:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.031189316883683205:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.005116013810038567:
                                                        return 't'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.038908498361706734:
                                            if Q.tau21 > 0.3683549016714096:
                                                return 'g'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.03278025798499584:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 48% of the training jets here get this class from the formula
                        else:
                            if Q.LHA > 0.26913610100746155:
                                if Q.pt_7 > 32.3125:
                                    if Q.mass > 27.44013786315918:
                                        if Q.girth2 > 0.003956989152356982:
                                            if Q.log_sum_pt > 5.941773176193237:
                                                return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 51% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 69% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.09665490686893463:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.004584567621350288:
                                            return 'W'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 100% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.015747987665235996:
                                    if Q.mass > 28.04977512359619:
                                        if Q.max_dr > 0.1601569503545761:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 48.453125:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 28.842333793640137:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.8943924307823181:
                                                        if Q.planar_flow > 0.10272857919335365:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.0513367410749197:
                                        if Q.centroid_offset > 0.003936836030334234:
                                            if Q.girth2 > 0.0038137363735586405:
                                                if Q.mass > 28.556401252746582:
                                                    if Q.pt_7 > 32.28125:
                                                        if Q.max_dr > 0.13664032518863678:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.05900937505066395:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.020398595370352268:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.008122436702251434:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.017738458700478077:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.0022954278392717242:
                                                    return 'g'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 48.0:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.02610096801072359:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 470.921875:
                                                if Q.centroid_offset > 0.011577699799090624:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.028991486877202988:
                            if Q.log_sum_pt > 6.379195213317871:
                                if Q.mass > 13.70293664932251:
                                    if Q.centroid_offset > 0.03669223003089428:
                                        if Q.centroid_offset > 0.04936378262937069:
                                            if Q.LHA > 0.27967777848243713:
                                                if Q.z_7 > 0.06433884799480438:
                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 26.1796875:
                                                if Q.max_dr > 0.0829155258834362:
                                                    if Q.LHA > 0.23990461230278015:
                                                        if Q.lam2 > 4.5457582018570974e-05:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 47% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 50.984375:
                                                        return 'g'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.21920569986104965:
                                            if Q.planar_flow > 0.2888606935739517:
                                                if Q.centroid_offset > 0.0329792071133852:
                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00040353514486923814:
                                                        if Q.LHA > 0.2297351285815239:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 473.453125:
                                                if Q.max_dr > 0.07342161983251572:
                                                    if Q.centroid_offset > 0.03231964260339737:
                                                        return 'Z'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.031885022297501564:
                                                        return 'Z'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.04014568775892258:
                                        if Q.lam1 > 0.005598593503236771:
                                            return 't'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.006586857372894883:
                                                if Q.lam1 > 0.0026276938151568174:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.06993420049548149:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.004636579658836126:
                                                    if Q.z_7 > 0.056698277592659:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.463833570480347:
                                            if Q.centroid_offset > 0.031116582453250885:
                                                if Q.z_7 > 0.08607619628310204:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.038843655958771706:
                                                        if Q.e2 > 0.005104662152007222:
                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 511.6875:
                                                    if Q.z_7 > 0.059029167518019676:
                                                        if Q.sum_pt > 680.953125:
                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.029710297472774982:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2_sq > 4.3168553020223044e-05:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 48% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.20882198214530945:
                                                if Q.mass > 4.949136734008789:
                                                    if Q.pt_7 > 28.265625:
                                                        if Q.log_sum_pt > 6.419969320297241:
                                                            if Q.sum_pt_top5 > 461.546875:
                                                                if Q.planar_flow > 0.10363643988966942:
                                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.D2 > 1.715740442276001:
                                                                        return 'g'   # 46% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth > 0.04083016328513622:
                                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p05_0p1 > 0.1416718065738678:
                                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.planar_flow > 0.14278768748044968:
                                                                        if Q.e2 > 0.00900750420987606:
                                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.LHA > 0.2166561335325241:
                                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'g'   # 50% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 624.8125:
                                                        if Q.girth2 > 0.0013829877716489136:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 502.875:
                                                    if Q.LHA > 0.20212027430534363:
                                                        if Q.z_7 > 0.04918937012553215:
                                                            return 'Z'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.03237685188651085:
                                                        if Q.log_sum_pt > 6.440323114395142:
                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.08094925805926323:
                                    if Q.log_sum_pt > 6.179445743560791:
                                        if Q.sum_pt_top5 > 408.390625:
                                            return 't'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.007883097510784864:
                                                return 't'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.295827865600586:
                                        if Q.mass > 17.562167167663574:
                                            if Q.centroid_offset > 0.03517149202525616:
                                                if Q.girth2 > 0.004150480730459094:
                                                    if Q.max_dr > 0.10165175423026085:
                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.4735099822282791:
                                                        if Q.planar_flow > 0.11451902613043785:
                                                            if Q.z_dr_0p05_0p1 > 0.6549210846424103:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.019160635769367218:
                                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 20.802045822143555:
                                                                if Q.pt_0 > 143.8125:
                                                                    return 'W'   # 45% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 46% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.24158302694559097:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 458.640625:
                                                        if Q.z_dr_0p1_0p2 > 0.015390267595648766:
                                                            return 'g'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p05_0p1 > 0.25719253718852997:
                                                if Q.centroid_offset > 0.04728882387280464:
                                                    if Q.centroid_offset > 0.07121571153402328:
                                                        if Q.pt_7 > 34.90625:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.47662244737148285:
                                                        if Q.LHA > 0.24001885205507278:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 8.000478192116134e-05:
                                                            if Q.centroid_offset > 0.03788425028324127:
                                                                if Q.girth > 0.046317657455801964:
                                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 22.86647605895996:
                                            if Q.e2 > 0.015925271436572075:
                                                if Q.sum_pt > 511.265625:
                                                    if Q.z_dr_0p05_0p1 > 0.5620786249637604:
                                                        if Q.centroid_offset > 0.03574712574481964:
                                                            return 'g'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 99% of the training jets here get this class from the formula
                        else:
                            if Q.D2 > 0.9209718406200409:
                                if Q.z_7 > 0.05760509334504604:
                                    if Q.centroid_offset > 0.024433540180325508:
                                        if Q.mass > 21.368672370910645:
                                            if Q.sum_pt_top5 > 445.828125:
                                                if Q.LHA > 0.2324845865368843:
                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 489.25:
                                                        return 'W'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.276143789291382:
                                                    if Q.width > 0.002532197395339608:
                                                        return 'W'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 525.34375:
                                                if Q.girth > 0.031046226620674133:
                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 483.25:
                                                    if Q.LHA > 0.20460522174835205:
                                                        if Q.eccentricity > 0.9012311398983002:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.003600003896281123:
                                            return 'g'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.0678117871284485:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 504.40625:
                                                    if Q.z_7 > 0.07917043566703796:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.008767419494688511:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 15.036952495574951:
                                                        if Q.z_7 > 0.07640668749809265:
                                                            return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 439.90625:
                                                                return 'q'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.3874175697565079:
                                        if Q.LHA > 0.211862251162529:
                                            if Q.sum_pt > 594.46875:
                                                return 'W'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 671.2734375:
                                                if Q.centroid_offset > 0.02570050861686468:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.00580298388376832:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.05120548605918884:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 61% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.01753405574709177:
                                                    if Q.mass > 11.165488243103027:
                                                        if Q.centroid_offset > 0.006744995014742017:
                                                            return 'g'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 1.407997727394104:
                                                                return 'g'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top5 > 496.53125:
                                                                    if Q.z_7 > 0.052388545125722885:
                                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 53% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.0066985993180423975:
                                            if Q.centroid_offset > 0.02246866002678871:
                                                if Q.log_sum_pt > 6.341281890869141:
                                                    if Q.LHA > 0.20873821526765823:
                                                        if Q.pt_7 > 27.046875:
                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 632.078125:
                                                            if Q.LHA > 0.19250938296318054:
                                                                return 'W'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 505.484375:
                                                    if Q.z_7 > 0.047090278938412666:
                                                        if Q.centroid_offset > 0.00792387081310153:
                                                            return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.17090946435928345:
                                                                return 'q'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.012695927172899246:
                                                            if Q.LHA > 0.19958845525979996:
                                                                return 'q'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.1721854731440544:
                                                                return 'q'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.5799981951713562:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.307941675186157:
                                                    if Q.mass > 12.024390697479248:
                                                        if Q.LHA > 0.17791704088449478:
                                                            return 'q'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 491.234375:
                                                                return 'q'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.005759026622399688:
                                    if Q.z_7 > 0.05724589340388775:
                                        if Q.centroid_offset > 0.02299394365400076:
                                            if Q.mass > 24.91685199737549:
                                                return 'W'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 469.34375:
                                                    if Q.LHA > 0.2113218605518341:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.007706743897870183:
                                                return 'g'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.06413286551833153:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.2201269418001175:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 63% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.01102375891059637:
                                            if Q.centroid_offset > 0.02298849355429411:
                                                if Q.log_sum_pt > 6.434394836425781:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 562.7734375:
                                                if Q.LHA > 0.17682035267353058:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 434.671875:
                                        if Q.z_7 > 0.07599728554487228:
                                            if Q.D2 > 0.6691285669803619:
                                                if Q.centroid_offset > 0.003201504237949848:
                                                    return 'g'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.08452108502388:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 0.809552401304245:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 12.252527713775635:
                                                if Q.centroid_offset > 0.004008942050859332:
                                                    if Q.D2 > 0.8106562495231628:
                                                        if Q.z_7 > 0.0657959133386612:
                                                            return 'g'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.004925778601318598:
                                                                return 'g'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 481.84375:
                                                    if Q.LHA > 0.13802470266819:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 0.6176607012748718:
                                            if Q.z_7 > 0.08036429807543755:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 0.7346032559871674:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.056996311992406845:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 58% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.08316771686077118:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 55% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
