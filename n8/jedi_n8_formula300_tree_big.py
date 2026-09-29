"""JEDI-linear jet tagger, 8 particles, 3 features: the formula with the fewest quantities (24) at the main result's accuracy (from the 931-term tuned formula): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 64.36% (the formula: 65.49%); same class as the formula for 91.19% of jets.  951 leaves, depth 17.
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
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        pt_7=pt[7],
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sum_pt_top5=sum(pt[:5]),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def decide(Q):
    if Q.girth2 > 0.009197166189551353:
        if Q.sum_pt > 882.578125:
            if Q.planar_flow > 0.23341487348079681:
                if Q.mass > 143.2337875366211:
                    return 'g'   # 60% of the training jets here get this class from the formula
                else:
                    return 't'   # 83% of the training jets here get this class from the formula
            else:
                if Q.mass > 125.33787536621094:
                    if Q.pt_7 > 16.4765625:
                        return 'g'   # 93% of the training jets here get this class from the formula
                    else:
                        return 'q'   # 44% of the training jets here get this class from the formula
                else:
                    if Q.pt_7 > 47.875:
                        return 'g'   # 88% of the training jets here get this class from the formula
                    else:
                        if Q.D2 > 0.4680263102054596:
                            if Q.pt_7 > 20.265625:
                                if Q.lam2 > 0.00011896226351382211:
                                    return 't'   # 46% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 80% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 55% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.832535028457642:
                                if Q.mass > 106.94198989868164:
                                    return 'g'   # 62% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 72% of the training jets here get this class from the formula
                            else:
                                return 't'   # 85% of the training jets here get this class from the formula
        else:
            if Q.mass > 45.29232215881348:
                if Q.girth2 > 0.009854927193373442:
                    if Q.lam1 > 0.03365671634674072:
                        if Q.eccentricity > 0.9877690374851227:
                            if Q.LHA > 0.4676365852355957:
                                if Q.pt_7 > 39.296875:
                                    return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 108.93879699707031:
                                        return 'q'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.04818134196102619:
                                            return 'g'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt > 0.2026512175798416:
                                                return 'g'   # 43% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 44.796875:
                                    return 'g'   # 71% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 82% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 128.13347625732422:
                                return 'g'   # 53% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.08647641912102699:
                                    if Q.max_pair_mass > 15.554545402526855:
                                        return 't'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 70% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 61.75417137145996:
                                        if Q.lam1 > 0.04416605830192566:
                                            if Q.C2 > 0.09272724017500877:
                                                return 't'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.08181274682283401:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 94% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 56% of the training jets here get this class from the formula
                    else:
                        if Q.lam2 > 0.0010651441989466548:
                            return 't'   # 99% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 117.481689453125:
                                if Q.pt_7 > 37.984375:
                                    if Q.D2 > 0.7309801280498505:
                                        return 't'   # 96% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.10793690383434296:
                                        return 't'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 127.1347427368164:
                                            return 'q'   # 49% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.023210606537759304:
                                                return 'q'   # 43% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.D2 > 0.6824099719524384:
                                    if Q.mass > 56.141014099121094:
                                        if Q.e2 > 0.03380902111530304:
                                            if Q.pt_7 > 46.234375:
                                                if Q.mass_over_sum_pt > 0.11827917769551277:
                                                    if Q.LHA > 0.4380158632993698:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.655641317367554:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.441783607006073:
                                                    if Q.e2 > 0.07116496935486794:
                                                        return 't'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.055261556059122086:
                                                        return 't'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 38.921875:
                                                            if Q.sum_pt > 777.890625:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.03555028885602951:
                                                return 't'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 35.390625:
                                                    if Q.log_sum_pt > 6.5932276248931885:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.03857515752315521:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.2766603827476501:
                                                        if Q.sum_pt_top5 > 581.125:
                                                            return 'q'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.015420511830598116:
                                            if Q.eccentricity > 0.9891487956047058:
                                                if Q.C2 > 0.03375632502138615:
                                                    if Q.e2 > 0.022964789532124996:
                                                        return 't'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.1864820420742035:
                                                    if Q.z_7 > 0.07888097316026688:
                                                        if Q.lam1 > 0.013511843513697386:
                                                            return 't'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 37.953125:
                                                if Q.mass_over_sum_pt > 0.10068292543292046:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 51.589895248413086:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 49% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.02717606071382761:
                                        if Q.e2 > 0.07424858957529068:
                                            return 't'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 38.34375:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 67.5625:
                                            return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 820.015625:
                                                if Q.pt_7 > 45.8125:
                                                    if Q.D2 > 0.2982875853776932:
                                                        return 'g'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 98% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.047675637528300285:
                        if Q.lam1 > 0.009444477036595345:
                            if Q.tau21 > 0.13352372497320175:
                                return 'Z'   # 64% of the training jets here get this class from the formula
                            else:
                                return 't'   # 78% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 32.765625:
                                return 'Z'   # 82% of the training jets here get this class from the formula
                            else:
                                return 't'   # 52% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.02446748036891222:
                            if Q.centroid_offset > 0.017421052791178226:
                                if Q.pt_7 > 50.59375:
                                    if Q.D2 > 1.1909882426261902:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.LHA > 0.2433253675699234:
                                    if Q.sum_pt > 804.171875:
                                        return 'Z'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 34.921875:
                                            if Q.max_pair_mass > 22.68217372894287:
                                                return 'Z'   # 51% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 94% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 53% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 52% of the training jets here get this class from the formula
            else:
                if Q.C2 > 0.04113775119185448:
                    if Q.centroid_offset > 0.03310978040099144:
                        if Q.sum_pt > 319.7890625:
                            if Q.centroid_offset > 0.07326674088835716:
                                return 't'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 38.3280029296875:
                                    if Q.eccentricity > 0.8661460876464844:
                                        if Q.max_dr > 0.2575816661119461:
                                            return 'g'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.3295360654592514:
                                        return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 64% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 64% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 40.02334976196289:
                            if Q.centroid_offset > 0.022960337810218334:
                                return 't'   # 54% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt > 0.09886714816093445:
                                    return 'g'   # 77% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.048775579780340195:
                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 73% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 88% of the training jets here get this class from the formula
                else:
                    if Q.tau21 > 0.16265378892421722:
                        if Q.width > 0.01065956661477685:
                            if Q.z_7 > 0.09621333703398705:
                                return 'g'   # 55% of the training jets here get this class from the formula
                            else:
                                return 't'   # 90% of the training jets here get this class from the formula
                        else:
                            return 't'   # 61% of the training jets here get this class from the formula
                    else:
                        if Q.log_sum_pt > 5.689354658126831:
                            return 't'   # 94% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 56% of the training jets here get this class from the formula
    else:
        if Q.mass > 30.15652084350586:
            if Q.width > 0.006653153337538242:
                if Q.centroid_offset > 0.028985573910176754:
                    if Q.max_dr > 0.12786699086427689:
                        if Q.D2 > 1.5762152671813965:
                            if Q.z_7 > 0.04687724635004997:
                                if Q.pt_7 > 37.609375:
                                    if Q.sum_pt_top5 > 520.109375:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.312394842505455:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.31972022354602814:
                                                return 'g'   # 46% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 77% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.2922957241535187:
                                        if Q.log_sum_pt > 6.217712640762329:
                                            return 't'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 70% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.491837024688721:
                                            return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.planar_flow > 0.06240415573120117:
                                    if Q.planar_flow > 0.47311684489250183:
                                        return 't'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.03951749950647354:
                                            return 't'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 61% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.789525747299194:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 88% of the training jets here get this class from the formula
                        else:
                            if Q.tau21 > 0.22276972979307175:
                                if Q.width > 0.007468974916264415:
                                    if Q.mass > 40.48733901977539:
                                        if Q.pt_7 > 39.109375:
                                            return 'g'   # 55% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 58% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.03834248334169388:
                                        return 't'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 73% of the training jets here get this class from the formula
                            else:
                                if Q.lam2 > 0.0001633848441997543:
                                    return 't'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.007644237717613578:
                                        if Q.log_sum_pt > 6.763054609298706:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.055718470364809036:
                                                return 'g'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.028032734990119934:
                                            if Q.mass > 41.91707992553711:
                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 69% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.040531009435653687:
                            if Q.tau21 > 0.1943352222442627:
                                if Q.sum_pt_top5 > 404.9375:
                                    return 't'   # 42% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.eccentricity > 0.9538301527500153:
                                    if Q.centroid_offset > 0.05625280924141407:
                                        return 'g'   # 66% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 65% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.0081490408629179:
                                if Q.centroid_offset > 0.03223488666117191:
                                    return 't'   # 75% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 55% of the training jets here get this class from the formula
                            else:
                                if Q.planar_flow > 0.2099568098783493:
                                    if Q.D2 > 0.7579529583454132:
                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 530.953125:
                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 32.625:
                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 82% of the training jets here get this class from the formula
                else:
                    if Q.girth2 > 0.006919040577486157:
                        if Q.max_dr > 0.23119552433490753:
                            if Q.sum_pt > 796.6640625:
                                if Q.centroid_offset > 0.013197268359363079:
                                    if Q.pt_7 > 38.359375:
                                        return 'g'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.3058164268732071:
                                            return 'q'   # 46% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 33.078125:
                                    if Q.width > 0.008187792729586363:
                                        if Q.C2 > 0.04416617192327976:
                                            return 'g'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.020060714334249496:
                                            return 't'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 507.59375:
                                        if Q.girth2 > 0.008034841157495975:
                                            return 't'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 2.327142357826233:
                                                return 't'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 667.0234375:
                                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 73% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 64% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 509.890625:
                                if Q.girth2 > 0.008778910152614117:
                                    if Q.e2 > 0.045539503917098045:
                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 66.00582504272461:
                                            if Q.centroid_offset > 0.01930929161608219:
                                                return 't'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 75% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.023933698423206806:
                                        if Q.max_dr > 0.16579648107290268:
                                            if Q.width > 0.007525353925302625:
                                                return 't'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.9476639628410339:
                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.17999900877475739:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.20332185178995132:
                                                        return 't'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.007027597166597843:
                                            if Q.log_sum_pt > 7.004067420959473:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 617.203125:
                                                    return 'Z'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 27.421875:
                                                        if Q.width > 0.007152078207582235:
                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.010882237460464239:
                                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.planar_flow > 0.04338277131319046:
                                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.e2 > 0.040604954585433006:
                                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.04323025047779083:
                                                            return 'Z'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.526266098022461:
                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.011969826649874449:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.05949254520237446:
                                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.03976680524647236:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 32.328125:
                                    if Q.e2 > 0.04386414960026741:
                                        if Q.girth2 > 0.008824712131172419:
                                            if Q.e2 > 0.04791068844497204:
                                                if Q.n_pt_above_50 > 2.5:
                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 48% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.008359836414456367:
                                            return 't'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.2197466418147087:
                                        if Q.pt_7 > 28.5390625:
                                            if Q.e2 > 0.044739535078406334:
                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 51% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 461.1171875:
                                                return 't'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 30.1796875:
                                            if Q.max_dr > 0.1094990149140358:
                                                return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 89% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.040062110871076584:
                            if Q.lam2 > 0.0002717693569138646:
                                return 'Z'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 64.19021606445312:
                                    if Q.width > 0.0067559340968728065:
                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.7634875774383545:
                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.010231659281998873:
                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.012282670009881258:
                                        if Q.sum_pt > 665.953125:
                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 0.6794883012771606:
                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.016844701021909714:
                                                    return 'Z'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.006796995643526316:
                                            if Q.sum_pt > 724.421875:
                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9750242531299591:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.14014177024364471:
                                                return 'Z'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 678.4140625:
                                if Q.D2 > 2.7334671020507812:
                                    return 't'   # 31% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.03869481198489666:
                                        if Q.sum_pt > 737.5859375:
                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.011355069931596518:
                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.z_dr_0p1_0p2 > 0.25963881611824036:
                                    if Q.centroid_offset > 0.01146986847743392:
                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 27.6484375:
                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 60% of the training jets here get this class from the formula
            else:
                if Q.max_dr > 0.15002717822790146:
                    if Q.girth2 > 0.004923145519569516:
                        if Q.centroid_offset > 0.011066232342272997:
                            if Q.centroid_offset > 0.03423669748008251:
                                if Q.sum_pt_top5 > 547.625:
                                    if Q.D2 > 2.6139602661132812:
                                        return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 40.3125:
                                            return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.005540018202736974:
                                                return 't'   # 37% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.width > 0.005798767786473036:
                                        if Q.e2 > 0.026829459704458714:
                                            return 'Z'   # 41% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 35.28125:
                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 57% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 557.484375:
                                    if Q.lam1 > 0.005364547250792384:
                                        if Q.D2 > 3.09802508354187:
                                            if Q.log_sum_pt > 6.57981538772583:
                                                return 'Z'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.2175055295228958:
                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 52.09213829040527:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p1_0p2 > 0.13542486727237701:
                                            if Q.centroid_offset > 0.016767913475632668:
                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 1.498756468296051:
                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.032582422718405724:
                                                if Q.width > 0.00551576865836978:
                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 4.320018291473389:
                                                    return 'Z'   # 36% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.11950185149908066:
                                                        if Q.centroid_offset > 0.01489343773573637:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 61.24084663391113:
                                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.19757282733917236:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 46% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 30.8359375:
                                        if Q.girth > 0.05098891817033291:
                                            if Q.centroid_offset > 0.014778312295675278:
                                                if Q.tau21 > 0.21801944822072983:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 34.421875:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.005985225550830364:
                                                    return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 37% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.38462334871292114:
                                            return 'g'   # 48% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 69% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.005849120439961553:
                                if Q.max_dr > 0.1673920825123787:
                                    if Q.mass > 56.11792182922363:
                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 29.9765625:
                                            if Q.max_dr > 0.19065653532743454:
                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.006200826959684491:
                                                    return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 39% of the training jets here get this class from the formula
                                else:
                                    if Q.width > 0.006186256418004632:
                                        if Q.mass > 53.97350883483887:
                                            if Q.e2 > 0.034633977338671684:
                                                if Q.width > 0.006428545108065009:
                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 59% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 67.27285385131836:
                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.00810063537210226:
                                                if Q.e2 > 0.03292067348957062:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.20662906020879745:
                                    if Q.max_dr > 0.24167203903198242:
                                        if Q.log_sum_pt > 6.480595350265503:
                                            if Q.centroid_offset > 0.00538908364251256:
                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.0053430425468832254:
                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 2.154688835144043:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 37% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.005449697840958834:
                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.008522720541805029:
                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00010971328447340056:
                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.18446098268032074:
                                        if Q.width > 0.005450092256069183:
                                            if Q.centroid_offset > 0.0062016891315579414:
                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 73.42508697509766:
                                            if Q.pt_7 > 18.03125:
                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.005593251436948776:
                                                if Q.e2 > 0.029708461835980415:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.005370822502300143:
                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 97% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.02011914551258087:
                            if Q.lam1 > 0.0033407355658710003:
                                if Q.max_dr > 0.17105618119239807:
                                    if Q.centroid_offset > 0.038021981716156006:
                                        if Q.pt_7 > 26.0625:
                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.916271209716797:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 36.78616905212402:
                                                if Q.max_dr > 0.3185844421386719:
                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.0036632869159802794:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p1_0p2 > 0.06368638202548027:
                                                            if Q.centroid_offset > 0.023606738075613976:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.0487921554595232:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.195685513317585:
                                                        return 'Z'   # 40% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.025739528238773346:
                                        if Q.LHA > 0.24794112890958786:
                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 47% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.004257509019225836:
                                            if Q.mass > 40.816497802734375:
                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.023671234026551247:
                                                if Q.mass > 38.76853561401367:
                                                    if Q.max_dr > 0.16196148842573166:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.2038055956363678:
                                    if Q.centroid_offset > 0.02325807698071003:
                                        if Q.pt_7 > 13.0234375:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.23455733805894852:
                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 41.4935359954834:
                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.027861222624778748:
                                        if Q.max_dr > 0.17443165183067322:
                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.0476871095597744:
                                                if Q.D2 > 2.7159037590026855:
                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 44% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.00023545446310890839:
                                            if Q.centroid_offset > 0.02408133912831545:
                                                if Q.mass > 35.75788116455078:
                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 45.22157859802246:
                                                return 'W'   # 51% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.02633048966526985:
                                                    if Q.mass > 37.16560935974121:
                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                        else:
                            if Q.girth > 0.02247863169759512:
                                if Q.max_dr > 0.22487571090459824:
                                    if Q.girth2 > 0.0032770195975899696:
                                        if Q.centroid_offset > 0.012668156065046787:
                                            if Q.girth2 > 0.0038355184951797128:
                                                if Q.mass > 36.69905662536621:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 40% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 3.501375436782837:
                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.01630992628633976:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.0036175540881231427:
                                                            if Q.e2 > 0.01397881330922246:
                                                                return 'W'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.05032227002084255:
                                                if Q.lam1 > 0.003909175982698798:
                                                    if Q.mass > 47.67571449279785:
                                                        if Q.centroid_offset > 0.005161971785128117:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.0602839607745409:
                                                                return 'Z'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 45% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 4.512627601623535:
                                                        if Q.LHA > 0.17425177246332169:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.3417930006980896:
                                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.004554663551971316:
                                                    if Q.centroid_offset > 0.009064281824976206:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.01773504912853241:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.0054092928767204285:
                                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 1090.62890625:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.0039935612585395575:
                                                            if Q.D2 > 2.4622074365615845:
                                                                if Q.centroid_offset > 0.006502195028588176:
                                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.010287422221153975:
                                                                    if Q.max_dr > 0.24727267771959305:
                                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.2745889574289322:
                                            if Q.LHA > 0.18459174036979675:
                                                if Q.lam1 > 0.00257700402289629:
                                                    if Q.D2 > 4.385047435760498:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.018244506791234016:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 40.61830520629883:
                                                    if Q.max_dr > 0.33078326284885406:
                                                        if Q.girth > 0.02817186899483204:
                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.779611110687256:
                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.018094578757882118:
                                                if Q.width > 0.002760825213044882:
                                                    if Q.max_dr > 0.24088570475578308:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 37.236371994018555:
                                                    if Q.sum_pt > 1133.1796875:
                                                        return 'Z'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.01400678837671876:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 59% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.016269882209599018:
                                        if Q.girth2 > 0.003979490837082267:
                                            if Q.max_dr > 0.18234644085168839:
                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.004596703685820103:
                                                    if Q.e2 > 0.02392602525651455:
                                                        return 'W'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.16320843994617462:
                                                        if Q.lam1 > 0.004395768046379089:
                                                            return 'Z'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 52.720157623291016:
                                                return 'Z'   # 50% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.00350590655580163:
                                                    if Q.max_dr > 0.20559190958738327:
                                                        if Q.centroid_offset > 0.01798874419182539:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.0022814639378339052:
                                            if Q.log_sum_pt > 6.968755006790161:
                                                if Q.pt_7 > 19.453125:
                                                    if Q.mass > 58.55902671813965:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 34.33606147766113:
                                                    if Q.centroid_offset > 0.013990374282002449:
                                                        if Q.mass_over_sum_pt > 0.06430003419518471:
                                                            if Q.max_dr > 0.18548548966646194:
                                                                return 'Z'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 26.1953125:
                                                        if Q.centroid_offset > 0.009251249488443136:
                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.07144073769450188:
                                                                return 'g'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.011479076463729143:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 39.13914489746094:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 25.1640625:
                                                        if Q.z_7 > 0.04740625433623791:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.width > 0.0022888047387823462:
                                    if Q.e2 > 0.009687494020909071:
                                        return 'W'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 60% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 16.8515625:
                                        if Q.mass > 39.19712829589844:
                                            if Q.girth > 0.02027896884828806:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 43% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.01069325814023614:
                                                return 'W'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 691.53125:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 94% of the training jets here get this class from the formula
                else:
                    if Q.width > 0.0026827623369172215:
                        if Q.centroid_offset > 0.024128050543367863:
                            if Q.girth2 > 0.005054939771071076:
                                if Q.max_dr > 0.11049194633960724:
                                    if Q.mass > 37.48317527770996:
                                        if Q.centroid_offset > 0.04120752029120922:
                                            return 't'   # 46% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.24784991145133972:
                                                return 't'   # 48% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.00535270432010293:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.1182580441236496:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.02830452285706997:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.23850250244140625:
                                            if Q.pt_7 > 31.5546875:
                                                if Q.centroid_offset > 0.04384548403322697:
                                                    return 'g'   # 45% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 46% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.03926275670528412:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 35.453125:
                                                    return 'Z'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.034838372841477394:
                                        if Q.centroid_offset > 0.04290865361690521:
                                            return 't'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 35.805137634277344:
                                                if Q.pt_7 > 28.046875:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 35.421875:
                                                    return 'Z'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.00605259882286191:
                                            if Q.log_sum_pt > 6.452221393585205:
                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.05451037362217903:
                                                    if Q.girth > 0.07832247018814087:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 43% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.10040628165006638:
                                                        if Q.e2 > 0.03832912817597389:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.00018399194959783927:
                                                if Q.C2 > 0.022336780093610287:
                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.03223733976483345:
                                    if Q.max_dr > 0.1209828071296215:
                                        if Q.LHA > 0.25535738468170166:
                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 7.521180305047892e-05:
                                            if Q.sum_pt > 796.0390625:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.1618928760290146:
                                                    if Q.D2 > 1.5672465562820435:
                                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.13111228495836258:
                                        if Q.LHA > 0.25775565207004547:
                                            if Q.centroid_offset > 0.02858134265989065:
                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.004685125779360533:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.00017870599549496546:
                                            if Q.tau21 > 0.13945560902357101:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 697.0:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.11026034504175186:
                                                        return 't'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.006298912689089775:
                                if Q.e2 > 0.03761158883571625:
                                    if Q.centroid_offset > 0.014368960168212652:
                                        if Q.D2 > 0.7514693737030029:
                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.03925515338778496:
                                                return 'W'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.566933870315552:
                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.006439464632421732:
                                                        return 'Z'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 72.86505126953125:
                                            if Q.pt_7 > 35.796875:
                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 81.32053756713867:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.008037216495722532:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 0.9181188344955444:
                                                if Q.width > 0.006398448487743735:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.03911479003727436:
                                                    if Q.lam2 > 0.000248797849053517:
                                                        if Q.girth2 > 0.00659163948148489:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.006548417964950204:
                                                        if Q.log_sum_pt > 6.648111820220947:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.009341557510197163:
                                        if Q.log_sum_pt > 6.3948962688446045:
                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 31.765625:
                                                if Q.D2 > 0.5787593126296997:
                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 58.25449180603027:
                                            if Q.lam1 > 0.006428509252145886:
                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.035888681188225746:
                                                    return 'W'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 93% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 35.38075065612793:
                                    if Q.log_sum_pt > 6.944674730300903:
                                        if Q.mass > 70.89373016357422:
                                            if Q.pt_7 > 20.9453125:
                                                if Q.mass > 86.67378997802734:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 80.38541793823242:
                                                    return 'Z'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1144.578125:
                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.016863473691046238:
                                            if Q.girth2 > 0.0057080325204879045:
                                                if Q.max_dr > 0.11577010527253151:
                                                    if Q.e2 > 0.03695087134838104:
                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 47.43505668640137:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.01877093780785799:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 58.82630157470703:
                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.1038910411298275:
                                                            if Q.girth2 > 0.0060637828428298235:
                                                                if Q.sum_pt_top5 > 501.0625:
                                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 55.10967445373535:
                                                    if Q.max_dr > 0.13533958792686462:
                                                        if Q.width > 0.005122634815052152:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 916.765625:
                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.13745103776454926:
                                                        if Q.width > 0.00519295409321785:
                                                            if Q.centroid_offset > 0.02035692147910595:
                                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 51% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.004794171778485179:
                                                                if Q.centroid_offset > 0.02155312430113554:
                                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.0032488349825143814:
                                                if Q.mass > 75.19317245483398:
                                                    if Q.pt_7 > 31.21875:
                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.00766060478053987:
                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.13193538039922714:
                                                        if Q.width > 0.005893743364140391:
                                                            if Q.centroid_offset > 0.01085355132818222:
                                                                if Q.e2 > 0.034461282193660736:
                                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 69.03750991821289:
                                                                    if Q.pt_7 > 29.5703125:
                                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.055895231664180756:
                                                    if Q.centroid_offset > 0.009651250671595335:
                                                        if Q.pt_7 > 52.546875:
                                                            if Q.planar_flow > 0.06158760003745556:
                                                                return 'g'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.988469123840332:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.0035766332875937223:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.008093005046248436:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 41.76688194274902:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_7 > 0.047400614246726036:
                                                                return 'W'   # 45% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 64% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.012573959771543741:
                                        if Q.pt_7 > 28.7734375:
                                            if Q.centroid_offset > 0.016060933470726013:
                                                return 'W'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.0595356747508049:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 30.914694786071777:
                                                        return 'W'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 502.7109375:
                                                if Q.pt_7 > 22.671875:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 38% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 46% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.003904737182892859:
                                            if Q.girth2 > 0.004197224974632263:
                                                if Q.pt_7 > 27.796875:
                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 33.65157699584961:
                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.11211007088422775:
                                                if Q.centroid_offset > 0.009260458871722221:
                                                    return 'W'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 37.171875:
                            if Q.centroid_offset > 0.014588729944080114:
                                if Q.log_sum_pt > 6.811536550521851:
                                    if Q.planar_flow > 0.04838228225708008:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 52.453125:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.planar_flow > 0.08510410040616989:
                                        if Q.pt_7 > 50.546875:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.0019360868609510362:
                                                return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.eccentricity > 0.9878338873386383:
                                    if Q.pt_7 > 44.359375:
                                        if Q.mass > 49.336952209472656:
                                            return 'g'   # 39% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.11716873198747635:
                                                return 'W'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 40.21653747558594:
                                            return 'W'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.551860809326172:
                                                if Q.sum_pt_top5 > 791.625:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.013288761489093304:
                                return 'W'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 40.75567817687988:
                                    if Q.log_sum_pt > 7.187972784042358:
                                        return 'g'   # 50% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 42.458099365234375:
                                            return 'W'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.006476972950622439:
                                                return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 31.8828125:
                                        if Q.lam2 > 7.274370364029892e-05:
                                            return 'g'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 836.5:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.088548693805933:
                                                    return 'W'   # 42% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.0519059207290411:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.010843636002391577:
                                            if Q.mass > 36.71893119812012:
                                                return 'W'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.962823867797852:
                                                if Q.centroid_offset > 0.005626929923892021:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 697.828125:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.0541316866874695:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
        else:
            if Q.z_7 > 0.04599064402282238:
                if Q.centroid_offset > 0.004021958913654089:
                    if Q.centroid_offset > 0.019990808330476284:
                        if Q.log_sum_pt > 6.435824632644653:
                            if Q.centroid_offset > 0.027593708597123623:
                                if Q.centroid_offset > 0.04686124436557293:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_over_sum_pt > 0.03166466951370239:
                                        if Q.lam2 > 9.932836110237986e-05:
                                            if Q.pt_7 > 48.921875:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.03533211536705494:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.20566990226507187:
                                                        if Q.e2 > 0.014538466930389404:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.03932188265025616:
                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.004943431820720434:
                                            if Q.pt_7 > 50.890625:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.031792595982551575:
                                                    if Q.log_sum_pt > 6.678715229034424:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 664.625:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.010415620636194944:
                                                                if Q.tau21 > 0.4237222671508789:
                                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 687.5:
                                                        if Q.pt_7 > 42.234375:
                                                            if Q.C2 > 0.009456971194595098:
                                                                return 'g'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.38009119033813477:
                                                            if Q.eccentricity > 0.9825180768966675:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 48% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 866.90625:
                                                if Q.mass > 3.9833141565322876:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.19345753639936447:
                                                    if Q.pt_7 > 31.5546875:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.549762487411499:
                                                        if Q.pt_7 > 36.03125:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.z_7 > 0.059063417837023735:
                                    if Q.planar_flow > 0.07573339343070984:
                                        if Q.C2 > 0.0057338878978043795:
                                            if Q.tau21 > 0.275690495967865:
                                                if Q.pt_7 > 47.890625:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.00780081725679338:
                                                        if Q.max_dr > 0.10145000368356705:
                                                            return 'W'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 685.8125:
                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 47.71875:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 52.828125:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 653.265625:
                                                    if Q.lam1 > 0.0006830725178588182:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 615.484375:
                                            if Q.C2 > 0.005065894452854991:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 659.9375:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.06855699419975281:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 804.140625:
                                        if Q.mass > 6.835665225982666:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.000524527917150408:
                                                if Q.sum_pt > 949.5625:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 46.890625:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 676.390625:
                                            if Q.centroid_offset > 0.025677194818854332:
                                                if Q.log_sum_pt > 6.619176864624023:
                                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.009242708794772625:
                                                    if Q.pt_7 > 42.109375:
                                                        return 'g'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.4069730043411255:
                                                            if Q.log_sum_pt > 6.564274072647095:
                                                                return 'W'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.194706030189991:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 24.859736442565918:
                                if Q.width > 0.004768843529745936:
                                    if Q.tau21 > 0.2945762425661087:
                                        if Q.width > 0.005646644392982125:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.030705426819622517:
                                                if Q.pt_7 > 29.2109375:
                                                    if Q.mass > 27.228017807006836:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0005004784325137734:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 36% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p1_0p2 > 0.20443029701709747:
                                            return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 27.640625:
                                                if Q.centroid_offset > 0.03021907713264227:
                                                    if Q.girth > 0.08312903344631195:
                                                        return 'g'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 34.859375:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.03522348031401634:
                                        if Q.width > 0.004298717016354203:
                                            if Q.e2 > 0.020592632703483105:
                                                return 'g'   # 33% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 2.1450917720794678:
                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0001026883655868005:
                                                    return 'Z'   # 37% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 26.992956161499023:
                                            if Q.pt_7 > 28.625:
                                                if Q.LHA > 0.21839284151792526:
                                                    if Q.D2 > 1.1112860441207886:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.24950071424245834:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.271254062652588:
                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.02333198394626379:
                                                if Q.C2 > 0.039647893980145454:
                                                    if Q.pt_7 > 31.4140625:
                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 68% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.335289716720581:
                                    if Q.LHA > 0.20855703949928284:
                                        if Q.centroid_offset > 0.04533599317073822:
                                            if Q.width > 0.007438685512170196:
                                                return 't'   # 44% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.033918073400855064:
                                                if Q.tau21 > 0.3780245929956436:
                                                    if Q.pt_7 > 39.328125:
                                                        if Q.e2 > 0.0063531044870615005:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.07045939564704895:
                                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.36925555765628815:
                                                    if Q.C2 > 0.03590710088610649:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.02392176166176796:
                                                        if Q.z_7 > 0.08150365576148033:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.18897829949855804:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.000776923552621156:
                                            return 't'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.225610017776489:
                                                if Q.LHA > 0.26097820699214935:
                                                    return 't'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                    else:
                        if Q.z_7 > 0.052114056423306465:
                            if Q.mass > 27.712697982788086:
                                if Q.centroid_offset > 0.015265846624970436:
                                    if Q.max_dr > 0.1183435246348381:
                                        if Q.pt_7 > 30.78125:
                                            return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 70% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.03414217382669449:
                                            return 'W'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 75% of the training jets here get this class from the formula
                                else:
                                    if Q.width > 0.004195475718006492:
                                        if Q.pt_7 > 30.6328125:
                                            if Q.girth2 > 0.0062750582583248615:
                                                return 'Z'   # 42% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.037449657917022705:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 48% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.00578910973854363:
                                    if Q.eccentricity > 0.9684995114803314:
                                        if Q.z_7 > 0.060670280829072:
                                            if Q.centroid_offset > 0.008329865988343954:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 461.59375:
                                                    if Q.D2 > 0.8555477559566498:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 52.90625:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.011612390633672476:
                                                if Q.centroid_offset > 0.018520083278417587:
                                                    if Q.pt_7 > 39.59375:
                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.06489761546254158:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 46.453125:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 610.984375:
                                                            if Q.mass > 12.625274658203125:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.05687852203845978:
                                            return 'g'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.43607082962989807:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.007052616216242313:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.12797090411186218:
                                                        return 'q'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.05878949724137783:
                                        if Q.tau21 > 0.2681667059659958:
                                            if Q.width > 3.981055306212511e-05:
                                                if Q.planar_flow > 0.2557109296321869:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 479.40625:
                                                        if Q.pt_7 > 48.328125:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 1.3776068091392517:
                                                                return 'g'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 678.5625:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 448.9375:
                                                if Q.pt_7 > 50.96875:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.0559258759021759:
                                                        return 'g'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.16227993369102478:
                                                            return 'q'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 51% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.3452843576669693:
                                            if Q.width > 4.7009756599436514e-05:
                                                if Q.sum_pt > 701.859375:
                                                    if Q.pt_7 > 44.578125:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.6027915477752686:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 50.234375:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 830.921875:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 621.546875:
                                                    if Q.C2 > 0.01899856887757778:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.12514912337064743:
                                                            return 'q'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 581.890625:
                                                                return 'q'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.007006986532360315:
                                if Q.eccentricity > 0.945704460144043:
                                    if Q.centroid_offset > 0.013029518537223339:
                                        if Q.centroid_offset > 0.017824160866439342:
                                            if Q.log_sum_pt > 6.612293004989624:
                                                return 'W'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 25.973868370056152:
                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.15940354764461517:
                                                if Q.sum_pt > 626.8125:
                                                    if Q.max_dr > 0.13423720002174377:
                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 667.0625:
                                                            return 'g'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 846.90625:
                                            return 'g'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 658.734375:
                                                if Q.LHA > 0.1263260692358017:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 759.609375:
                                                        return 'q'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_pair_mass > 7.373898983001709:
                                                    return 'q'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.017428171820938587:
                                        if Q.log_sum_pt > 6.562841415405273:
                                            if Q.pt_7 > 41.734375:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 25.33764934539795:
                                                return 'W'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 11.83219289779663:
                                            if Q.lam2 > 9.231163494405337e-05:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 40.140625:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 665.25:
                                                        if Q.LHA > 0.14491969347000122:
                                                            if Q.centroid_offset > 0.012487797532230616:
                                                                return 'g'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.60502290725708:
                                                if Q.pt_7 > 39.984375:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.008763334713876247:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.5343454778194427:
                                                            return 'g'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.52039098739624:
                                    if Q.pt_7 > 42.546875:
                                        if Q.width > 4.657934550778009e-05:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 48.078125:
                                                return 'g'   # 100% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.609265089035034:
                                            if Q.centroid_offset > 0.005702537717297673:
                                                if Q.planar_flow > 0.558432400226593:
                                                    if Q.pt_7 > 39.359375:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.642670154571533:
                                                            return 'q'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.11534136533737183:
                                                return 'q'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.37582917511463165:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.08306974545121193:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.380143165588379:
                                                return 'q'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top5 > 527.421875:
                        if Q.pt_7 > 49.171875:
                            if Q.mass > 5.134068489074707:
                                if Q.centroid_offset > 0.0018446914036758244:
                                    if Q.sum_pt > 809.796875:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.41740652918815613:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.012053206097334623:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.07699104398488998:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.0033501420402899384:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 950.515625:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 6.536585330963135:
                                            if Q.log_sum_pt > 6.710903644561768:
                                                if Q.lam2 > 1.152011145677534e-05:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 53.34375:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.5651409924030304:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.01316126435995102:
                                                        return 'g'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.890895366668701:
                                    if Q.log_sum_pt > 6.935370445251465:
                                        return 'g'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.0015646509127691388:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 4.4428898036130704e-05:
                                        if Q.width > 4.980146150046494e-05:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.0032054662005975842:
                                            if Q.mass > 4.592646837234497:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 99% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top5 > 572.34375:
                                if Q.sum_pt > 966.25:
                                    if Q.width > 4.394032839627471e-05:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.0027941183652728796:
                                        if Q.pt_7 > 46.140625:
                                            if Q.lam2 > 1.658857217989862e-05:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.609454393386841:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.1036817692220211:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 4.783903386851307e-05:
                                                        return 'g'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.D2 > 1.3700613975524902:
                                    if Q.width > 4.134932351007592e-05:
                                        if Q.planar_flow > 0.5389090180397034:
                                            if Q.centroid_offset > 0.0010823169141076505:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 54% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 561.890625:
                                                return 'q'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0020751357078552246:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 65% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.planar_flow > 0.5663161277770996:
                                        if Q.girth2 > 4.437319512362592e-05:
                                            return 'g'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 95% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 91% of the training jets here get this class from the formula
                    else:
                        if Q.D2 > 1.0521426796913147:
                            if Q.girth2 > 3.6435307265492156e-05:
                                if Q.sum_pt_top5 > 484.015625:
                                    if Q.eccentricity > 0.8900289535522461:
                                        if Q.D2 > 1.4850645065307617:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.00208652438595891:
                                                if Q.pt_7 > 50.296875:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 79% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 661.859375:
                                    return 'q'   # 85% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 76% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top5 > 438.390625:
                                if Q.z_7 > 0.07791389152407646:
                                    if Q.centroid_offset > 0.002250904683023691:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 69% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.07154332473874092:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.34042833745479584:
                                            return 'g'   # 61% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 92% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 89% of the training jets here get this class from the formula
            else:
                if Q.centroid_offset > 0.015976176597177982:
                    if Q.centroid_offset > 0.02553854137659073:
                        if Q.sum_pt > 700.6796875:
                            if Q.log_sum_pt > 6.638854026794434:
                                if Q.log_sum_pt > 6.846237421035767:
                                    if Q.log_sum_pt > 6.914327144622803:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 50% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 22.63072109222412:
                                        if Q.centroid_offset > 0.030336924828588963:
                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.050544265657663345:
                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.24305275082588196:
                                            return 'g'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.027144379913806915:
                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.685080289840698:
                                                    if Q.n_pt_above_50 > 3.5:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 66% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.030141460709273815:
                                    if Q.pt_7 > 24.53125:
                                        if Q.centroid_offset > 0.04540732502937317:
                                            return 'q'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 22.070809364318848:
                                                if Q.lam2 > 3.5985620343126357e-05:
                                                    return 'Z'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 23.6015625:
                                        if Q.centroid_offset > 0.028847570531070232:
                                            if Q.lam1 > 0.0014245268539525568:
                                                return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p1_0p2 > 0.022541537880897522:
                                            return 'W'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 76% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 22.056960105895996:
                                if Q.log_sum_pt > 6.429177522659302:
                                    if Q.pt_7 > 23.4140625:
                                        if Q.centroid_offset > 0.039831919595599174:
                                            return 'q'   # 47% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 77% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.0033882393036037683:
                                        return 't'   # 66% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 56% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 660.5546875:
                                    if Q.z_7 > 0.036262672394514084:
                                        if Q.centroid_offset > 0.03227784112095833:
                                            if Q.centroid_offset > 0.045311423018574715:
                                                return 'q'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 27.3671875:
                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.005235663615167141:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 42% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.035279689356684685:
                                        if Q.width > 0.005737975472584367:
                                            return 'q'   # 39% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 520.8984375:
                                            return 'q'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                    else:
                        if Q.log_sum_pt > 6.5641419887542725:
                            if Q.centroid_offset > 0.018392334692180157:
                                if Q.sum_pt > 1004.9296875:
                                    if Q.pt_7 > 28.7421875:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 49% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.685324430465698:
                                        if Q.centroid_offset > 0.023858812637627125:
                                            if Q.sum_pt > 865.14453125:
                                                if Q.C2 > 0.008802246768027544:
                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 38.796875:
                                                if Q.centroid_offset > 0.02252264227718115:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 26.5078125:
                                            if Q.centroid_offset > 0.020241315476596355:
                                                return 'W'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 30.4609375:
                                                    if Q.log_sum_pt > 6.598987817764282:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.8911074697971344:
                                                        return 'q'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.001399654836859554:
                                                return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.021101702004671097:
                                                    if Q.pt_7 > 22.765625:
                                                        if Q.sum_pt > 744.203125:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.740611791610718:
                                    if Q.sum_pt > 1011.9765625:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 20.6953125:
                                            if Q.pt_7 > 36.453125:
                                                if Q.mass > 6.18717622756958:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 873.2578125:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.017242289148271084:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.030148563906550407:
                                                            return 'W'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.7923970222473145:
                                                if Q.centroid_offset > 0.01691756211221218:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.1430886685848236:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.0370310191065073:
                                        if Q.planar_flow > 0.3366519808769226:
                                            if Q.tau21 > 0.27472928166389465:
                                                if Q.sum_pt > 739.9765625:
                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.70550537109375:
                                                return 'W'   # 50% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 632.625:
                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.15347933024168015:
                                                        return 'q'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 65% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 24.462997436523438:
                                            return 'W'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 88% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top5 > 550.765625:
                                if Q.LHA > 0.17371054738759995:
                                    if Q.z_7 > 0.036623019725084305:
                                        if Q.lam1 > 0.0014853625907562673:
                                            return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.9212594330310822:
                                                if Q.centroid_offset > 0.022756511345505714:
                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.03038492053747177:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 80% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 23.069966316223145:
                                    if Q.pt_7 > 25.6015625:
                                        return 'W'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 7.930442370707169e-05:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 42% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 94% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top5 > 593.015625:
                        if Q.pt_7 > 38.546875:
                            if Q.log_sum_pt > 6.898320198059082:
                                if Q.girth > 0.00514754094183445:
                                    return 'g'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1087.84375:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.0062101553194224834:
                                    if Q.lam2 > 7.80478058004519e-06:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 54% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 951.0:
                                        if Q.centroid_offset > 0.0033324528485536575:
                                            if Q.girth2 > 4.6088292947388254e-05:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 93% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 98% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 1098.32421875:
                                if Q.pt_7 > 23.59375:
                                    if Q.centroid_offset > 0.0028771229553967714:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 30.0703125:
                                            if Q.mass > 6.577735424041748:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 1286.953125:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 7.243585586547852:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.007414817111566663:
                                        if Q.pt_7 > 16.8359375:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.007026660954579711:
                                    if Q.pt_7 > 28.6640625:
                                        if Q.sum_pt > 951.6640625:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.618937253952026:
                                                if Q.pt_7 > 35.734375:
                                                    if Q.lam2 > 2.1311917407729197e-05:
                                                        if Q.centroid_offset > 0.008444757666438818:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.013161097653210163:
                                                        if Q.log_sum_pt > 6.796317100524902:
                                                            if Q.centroid_offset > 0.014219239354133606:
                                                                return 'W'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.4599115699529648:
                                                                if Q.tau21 > 0.32221856713294983:
                                                                    if Q.pt_7 > 33.9375:
                                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 771.4921875:
                                                            return 'q'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.4455368369817734:
                                                                if Q.z_7 > 0.04365699738264084:
                                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 10.079678535461426:
                                                    if Q.eccentricity > 0.8196896016597748:
                                                        if Q.LHA > 0.1355261281132698:
                                                            return 'q'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 629.3203125:
                                            if Q.sum_pt > 1000.13671875:
                                                if Q.pt_7 > 22.0546875:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.014610985293984413:
                                                    if Q.sum_pt > 890.1875:
                                                        if Q.pt_7 > 20.5546875:
                                                            if Q.sum_pt > 918.421875:
                                                                return 'W'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.1466602087020874:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.593590021133423:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.026355321519076824:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 617.40625:
                                        if Q.sum_pt > 1032.6552734375:
                                            if Q.z_7 > 0.027873060666024685:
                                                if Q.width > 4.670401904149912e-05:
                                                    if Q.centroid_offset > 0.002304280176758766:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.004577345680445433:
                                                    if Q.pt_7 > 24.140625:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 100% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.005117305088788271:
                                            if Q.LHA > 0.11533218994736671:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.597060441970825:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 96% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.00447882479056716:
                            if Q.mass > 11.821224212646484:
                                if Q.lam2 > 6.551515616592951e-05:
                                    if Q.sum_pt_top5 > 550.765625:
                                        if Q.LHA > 0.16977856308221817:
                                            return 'q'   # 55% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 526.34375:
                                        if Q.LHA > 0.12496979907155037:
                                            return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 71% of the training jets here get this class from the formula
                            else:
                                if Q.z_7 > 0.023447257466614246:
                                    if Q.sum_pt > 691.609375:
                                        if Q.mass > 7.308181524276733:
                                            return 'q'   # 51% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 70% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top5 > 541.21875:
                                if Q.D2 > 1.610757827758789:
                                    if Q.z_7 > 0.03136777877807617:
                                        if Q.sum_pt > 677.8828125:
                                            return 'q'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 96% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 94% of the training jets here get this class from the formula
                            else:
                                if Q.z_7 > 0.02429694402962923:
                                    if Q.planar_flow > 0.4448729306459427:
                                        return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 62% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 76% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
