"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.6% (the network: 65.8%); same class as the network for 87.0% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.centroid_offset        distance of the pT centroid from the jet axis
"""
import math
from types import SimpleNamespace

CLASSES = ['g', 'q', 'W', 'Z', 't']
W = [[-0.15625, 0.0, 0.34375, 0.0, 0.015625], [0.390625, 0.0, -0.03125, 0.125, 0.0], [0.4296875, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.5, -0.5625, 0.0625], [-0.03125, -0.09375, 0.0, 0.078125, 0.125], [-0.1875, 0.046875, 0.0, 0.015625, -0.25], [0.109375, 0.125, -0.3125, -0.375, 0.0], [0.0, 0.0, 0.21875, 0.46875, 0.0], [0.0, 0.0625, -0.25, 0.0, 0.1875], [0.171875, 0.25390625, -0.03125, -0.03125, 0.0], [0.0, -0.125, 0.0, 0.0, 0.375], [0.0, 0.0, 0.375, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -0.5], [0.0, 0.0, 0.0703125, 0.0546875, -0.40625], [0.0, 0.0, -0.75, 0.375, 0.0], [0.0, 0.0625, -0.6875, -0.15625, 0.0]]
B = [-0.4375, 0.03125, -0.125, -0.09375, 1.34375]
INT_BITS = [3, 5, 4, 4, 5, 5, 4, 4, 3, 4, 5, 4, 5, 4, 3, 3]
FRAC_BITS = [3, 3, 4, 3, 2, 3, 3, 3, 3, 2, 4, 4, 3, 3, 4, 3]


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

    _memo = {}

    def m4(v):
        return math.sqrt(max(v[3] ** 2 - v[0] ** 2 - v[1] ** 2 - v[2] ** 2, 0.0))

    def ecf(name):
        if 'ecf' not in _memo:
            h3, h4 = min(n, 24), min(n, 12)
            Rm = {(i, j): math.sqrt(dist2(i, j)) for i in range(h3) for j in range(h3)}
            o = dict(e2=0.0, e2b2=0.0, e3=0.0, e3b2=0.0, g31=0.0, g32=0.0, e4=0.0, g41=0.0, g42=0.0)
            for i in range(h3):
                for j in range(i + 1, h3):
                    w = z[i] * z[j]
                    o['e2'] += w * Rm[i, j]
                    o['e2b2'] += w * Rm[i, j] ** 2
                    for k in range(j + 1, h3):
                        a, b, c = Rm[i, j], Rm[i, k], Rm[j, k]
                        w3 = z[i] * z[j] * z[k]
                        s = sorted((a, b, c))
                        o['e3'] += w3 * a * b * c
                        o['e3b2'] += w3 * (a * b * c) ** 2
                        o['g31'] += w3 * s[0]
                        o['g32'] += w3 * s[0] * s[1]
            for i in range(h4):
                for j in range(i + 1, h4):
                    for k in range(j + 1, h4):
                        for l in range(k + 1, h4):
                            d = (Rm[i, j], Rm[i, k], Rm[i, l], Rm[j, k], Rm[j, l], Rm[k, l])
                            w4 = z[i] * z[j] * z[k] * z[l]
                            s = sorted(d)
                            o['e4'] += w4 * d[0] * d[1] * d[2] * d[3] * d[4] * d[5]
                            o['g41'] += w4 * s[0]
                            o['g42'] += w4 * s[0] * s[1]
            _memo['ecf'] = o
        return _memo['ecf'][name]

    def kaxes(k):
        if ('ax', k) not in _memo:
            hard = max(P, key=lambda i: pt[i])
            axes = [(eta[hard], phi[hard])]
            for _ in range(1, k):
                far = max(P, key=lambda i: min(math.hypot(eta[i] - a, phi[i] - b) for a, b in axes) * pt[i])
                axes.append((eta[far], phi[far]))
            for _ in range(6):
                near = [min(range(k), key=lambda j: math.hypot(eta[i] - axes[j][0], phi[i] - axes[j][1])) for i in P]
                for j in range(k):
                    w = sum(pt[i] for i in P if near[i] == j)
                    if w > 0:
                        axes[j] = (sum(pt[i] * eta[i] for i in P if near[i] == j) / w, sum(pt[i] * phi[i] for i in P if near[i] == j) / w)
            dist = [[math.hypot(eta[i] - a, phi[i] - b) for a, b in axes] for i in P]
            _memo['ax', k] = (axes, dist)
        return _memo['ax', k]

    def tau_n(k, beta=1):
        axes, dist = kaxes(k)
        return sum(z[i] * min(dist[i]) ** beta for i in P) / 0.8 ** beta

    def subjets(k):
        if ('sj', k) not in _memo:
            axes, dist = kaxes(k)
            S, V = [0.0] * k, [[0.0] * 4 for _ in range(k)]
            for i in real:
                j = min(range(k), key=lambda a: dist[i][a])
                S[j] += z[i]
                for c, x in enumerate((pt[i] * math.cos(phi[i]), pt[i] * math.sin(phi[i]), pt[i] * math.sinh(eta[i]), pt[i] * math.cosh(eta[i]))):
                    V[j][c] += x
            order = sorted(range(k), key=lambda j: -S[j])
            pairs = [(p, q) for p in range(k) for q in range(p + 1, k)]
            _memo['sj', k] = dict(z=[S[j] for j in order], mass=[m4(V[j]) for j in order],
                                  dr=[math.hypot(axes[order[p]][0] - axes[order[q]][0], axes[order[p]][1] - axes[order[q]][1]) for p, q in pairs],
                                  mpair=[m4([V[order[p]][c] + V[order[q]][c] for c in range(4)]) for p, q in pairs])
        return _memo['sj', k]

    def softdrop(what):
        if 'sd' not in _memo:
            node = {i: (pt[i] * math.cos(phi[i]), pt[i] * math.sin(phi[i]), pt[i] * math.sinh(eta[i]), pt[i] * math.cosh(eta[i])) for i in range(min(n, 20)) if pt[i] > 0}
            live, kids, new = list(node), {}, 1000

            def dR2(u, v):
                ya, pa = 0.5 * math.log(max(u[3] + u[2], 1e-300) / max(u[3] - u[2], 1e-300)), math.atan2(u[1], u[0])
                yb, pb = 0.5 * math.log(max(v[3] + v[2], 1e-300) / max(v[3] - v[2], 1e-300)), math.atan2(v[1], v[0])
                return (ya - yb) ** 2 + ((pa - pb + math.pi) % (2 * math.pi) - math.pi) ** 2
            while len(live) > 1:                                  # Cambridge/Aachen: merge the closest pair
                best = None
                for a in range(len(live)):
                    for b in range(a + 1, len(live)):
                        d = dR2(node[live[a]], node[live[b]])
                        if best is None or d < best[0]:
                            best = (d, a, b)
                _, a, b = best
                node[new] = tuple(x + y for x, y in zip(node[live[a]], node[live[b]]))
                kids[new] = (live[a], live[b])
                live[a] = new
                live.pop(b)
                new += 1
            cur, removed, res = live[0], 0, (0.0, 0.0, 0.0)
            while cur in kids:                                    # soft drop, beta = 0, z_cut = 0.1
                c1, c2 = kids[cur]
                p1, p2 = math.hypot(node[c1][0], node[c1][1]), math.hypot(node[c2][0], node[c2][1])
                zz = min(p1, p2) / max(p1 + p2, 1e-300)
                if zz > 0.1:
                    res = (m4(node[cur]), zz, math.sqrt(dR2(node[c1], node[c2])))
                    break
                cur = c1 if p1 >= p2 else c2
                removed += 1
            _memo['sd'] = dict(mass=res[0], zg=res[1], rg=res[2], removed=float(removed))
        return _memo['sd'][what]

    def softp(s, what):
        j = len(real) - s
        if j < 15:
            return 0.0
        return dict(pt=pt[j], z=z[j], abseta=abs(eta[j]), absphi=abs(phi[j]), dr=dr[j], dr0=math.sqrt(dist2(0, j)))[what]

    def ncum(f):
        c = 0.0
        for k, i in enumerate(real):
            c += z[i]
            if c >= f:
                return float(k + 1)
        return float(len(real))

    return SimpleNamespace(
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_6=z[6],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        z_top3_slots=sum(pt[:3]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
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
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -1.770391
    if Q.planar_flow < 0.1484197:
        z += -11.94981 * Q.planar_flow + 1.773588
    if Q.lam1 < 0.004183811:
        z += 704.5015 * Q.lam1 - 0.7195551
    if 0.004183811 <= Q.lam1 < 0.00595415:
        z += -132.2298 * Q.lam1 + 2.78117
    if 0.00595415 <= Q.lam1 < 0.01200373:
        z += -329.5858 * Q.lam1 + 3.956258
    if Q.tau1 < 0.05356915:
        z += -28.857 * Q.tau1 + 1.545845
    if 0.1426152 <= Q.sj3_dr_max < 0.169029:
        z += 34.89595 * Q.sj3_dr_max - 4.976692
    if 0.169029 <= Q.sj3_dr_max < 0.1986272:
        z += 11.19242 * Q.sj3_dr_max - 0.9701064
    if Q.sj3_dr_max >= 0.1986272:
        z += -15.70748 * Q.sj3_dr_max + 4.372944
    if Q.centroid_offset >= 0.006789738:
        z += -98.43912 * Q.centroid_offset + 0.6683758
    if 6.766778 <= Q.log_sum_pt < 6.842717:
        z += -13.30302 * Q.log_sum_pt + 90.01861
    if Q.log_sum_pt >= 6.842717:
        z += -41.26115 * Q.log_sum_pt + 281.3282
    if Q.sum_pt_top5 >= 791.125:
        z += 0.01106801 * Q.sum_pt_top5 - 8.756181
    if Q.tau2 < 0.01713288:
        z += 215.1507 * Q.tau2 - 3.686151
    if 0.1546891 <= Q.LHA < 0.3033137:
        z += 11.22225 * Q.LHA - 1.73596
    if Q.LHA >= 0.3033137:
        z += -40.81292 * Q.LHA + 14.04702
    if Q.e3 < 0.0005116989:
        z += -3060.0 * Q.e3 + 1.565799
    if Q.sum_pt < 763.825 and Q.centroid_offset > 0.01096064:
        z += 0.2871354 * (763.825 - Q.sum_pt) * (Q.centroid_offset - 0.01096064)
    if Q.lam1 < 0.004183811 and Q.sum_pt < 763.825:
        z += 3.187821 * (0.004183811 - Q.lam1) * (763.825 - Q.sum_pt)
    if Q.sum_pt < 763.825 and Q.e3 < 0.0005116989:
        z += -24.56665 * (763.825 - Q.sum_pt) * (0.0005116989 - Q.e3)
    if Q.lam1 < 0.004183811 and Q.D2 < 0.7459513:
        z += -7498.787 * (0.004183811 - Q.lam1) * (0.7459513 - Q.D2)
    if Q.sum_pt < 763.825 and Q.z_7 > 0.01685855:
        z += 0.06094081 * (763.825 - Q.sum_pt) * (Q.z_7 - 0.01685855)
    if Q.planar_flow < 0.04505724 and Q.tau21_b2 < 0.05932655:
        z += 1136.469 * (0.04505724 - Q.planar_flow) * (0.05932655 - Q.tau21_b2)
    if Q.log_sum_pt > 6.842717 and Q.tau21_b2 < 0.04019753:
        z += -859.0172 * (Q.log_sum_pt - 6.842717) * (0.04019753 - Q.tau21_b2)
    if Q.sum_pt_top5 > 791.125 and Q.tau21_b2 < 0.1230713:
        z += 0.2425327 * (Q.sum_pt_top5 - 791.125) * (0.1230713 - Q.tau21_b2)
    if Q.e3 < 8.147744e-05 and Q.sj3_z3 < 0.0887688:
        z += 258447.3 * (8.147744e-05 - Q.e3) * (0.0887688 - Q.sj3_z3)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.329276
    if Q.lam1 < 0.002464291:
        z += -314.6675 * Q.lam1 - 0.319241
    if 0.002464291 <= Q.lam1 < 0.005433361:
        z += 56.50537 * Q.lam1 - 1.233919
    if 0.005433361 <= Q.lam1 < 0.008375572:
        z += 315.0368 * Q.lam1 - 2.638614
    if Q.pt_7 >= 34.53125:
        z += 0.1334208 * Q.pt_7 - 4.607186
    if 6.377723 <= Q.log_sum_pt < 6.572938:
        z += 5.632469 * Q.log_sum_pt - 35.92233
    if Q.log_sum_pt >= 6.572938:
        z += 16.90421 * Q.log_sum_pt - 110.0108
    if Q.z_7 < 0.06473447:
        z += 59.43783 * Q.z_7 - 3.847676
    if Q.tau1 < 0.1027642:
        z += -21.73305 * Q.tau1 + 2.233381
    if Q.sj3_dr_min >= 0.02913153:
        z += -15.71103 * Q.sj3_dr_min + 0.4576862
    if Q.e3 < 1.340118e-05:
        z += 65734.96 * Q.e3 - 0.8809263
    if Q.sj2_dr < 0.1591713:
        z += -15.40297 * Q.sj2_dr + 2.451711
    if Q.max_dr < 0.1117619:
        z += 18.5295 * Q.max_dr - 2.070891
    if Q.sum_pt_top5 >= 839.9547:
        z += 0.02196749 * Q.sum_pt_top5 - 18.4517
    if Q.z_6 < 0.04737278:
        z += 102.2767 * Q.z_6 - 4.84513
    if Q.log_sum_pt > 6.377723 and Q.e3 < 1.340118e-05:
        z += 410789.2 * (Q.log_sum_pt - 6.377723) * (1.340118e-05 - Q.e3)
    if Q.pt_7 > 34.53125 and Q.tau2 < 0.01713288:
        z += -7.605745 * (Q.pt_7 - 34.53125) * (0.01713288 - Q.tau2)
    if Q.lam1 < 0.008375572 and Q.lam2 < 0.001130645:
        z += -225060.9 * (0.008375572 - Q.lam1) * (0.001130645 - Q.lam2)
    if Q.lam1 < 0.01200373 and Q.n_pt_above_50 > 6.0:
        z += -107.9082 * (0.01200373 - Q.lam1) * (Q.n_pt_above_50 - 6.0)
    if Q.lam1 < 0.005433361 and Q.tau21_b2 > 0.04019753:
        z += -325.0313 * (0.005433361 - Q.lam1) * (Q.tau21_b2 - 0.04019753)
    if Q.z_7 < 0.06473447 and Q.girth2_top2 < 0.005180665:
        z += -10378.78 * (0.06473447 - Q.z_7) * (0.005180665 - Q.girth2_top2)
    if Q.z_6 < 0.04737278 and Q.girth2_top2 < 0.004007842:
        z += 21964.09 * (0.04737278 - Q.z_6) * (0.004007842 - Q.girth2_top2)
    if Q.log_sum_pt > 6.572938 and Q.girth2_top3 < 0.006756161:
        z += -2770.331 * (Q.log_sum_pt - 6.572938) * (0.006756161 - Q.girth2_top3)
    if Q.log_sum_pt > 6.572938 and Q.dr_max_012 < 0.1206357:
        z += 60.23992 * (Q.log_sum_pt - 6.572938) * (0.1206357 - Q.dr_max_012)
    if Q.n_pt_above_50 > 6.0 and Q.e3 < 8.147744e-05:
        z += 9980.256 * (Q.n_pt_above_50 - 6.0) * (8.147744e-05 - Q.e3)
    if Q.sum_pt > 988.4078 and Q.girth2_top2 < 0.0095303:
        z += -2.652052 * (Q.sum_pt - 988.4078) * (0.0095303 - Q.girth2_top2)
    if Q.sum_pt_top5 > 839.9547 and Q.e3 < 3.892127e-05:
        z += -435.0901 * (Q.sum_pt_top5 - 839.9547) * (3.892127e-05 - Q.e3)
    if Q.lam1 < 0.005433361 and Q.dr_max_012 < 0.1550922:
        z += -3271.932 * (0.005433361 - Q.lam1) * (0.1550922 - Q.dr_max_012)
    if Q.z_7 < 0.06473447 and Q.girth2_top3 < 0.007929074:
        z += 7040.998 * (0.06473447 - Q.z_7) * (0.007929074 - Q.girth2_top3)
    return max(0.0, z)


def neuron_2(Q):
    z = 5.020448
    if Q.M2 < 0.1222545:
        z += 17.10167 * Q.M2 - 2.090756
    if Q.pt_7 < 20.125:
        z += 0.0618162 * Q.pt_7 - 5.928629
    if 20.125 <= Q.pt_7 < 53.4375:
        z += 0.1465996 * Q.pt_7 - 7.634895
    if Q.pt_7 >= 53.4375:
        z += 0.03565452 * Q.pt_7 - 1.706266
    if Q.log_sum_pt < 6.605974:
        z += -10.72741 * Q.log_sum_pt + 71.38481
    if 6.605974 <= Q.log_sum_pt < 6.701242:
        z += -5.45642 * Q.log_sum_pt + 36.56479
    if Q.z_7 < 0.02320757:
        z += -30.72555 * Q.z_7 + 1.800938
    if 0.02320757 <= Q.z_7 < 0.02807091:
        z += -140.4506 * Q.z_7 + 4.34739
    if 0.02807091 <= Q.z_7 < 0.0586137:
        z += -99.09614 * Q.z_7 + 3.186532
    if Q.z_7 >= 0.0586137:
        z += -68.37059 * Q.z_7 + 1.385594
    if Q.girth < 0.007673833:
        z += 556.2368 * Q.girth - 1.078335
    if 0.007673833 <= Q.girth < 0.1019409:
        z += -33.84143 * Q.girth + 3.449827
    if Q.sum_pt >= 988.4078:
        z += -0.01194689 * Q.sum_pt + 11.8084
    if Q.sum_pt_top5 < 752.1:
        z += 0.002211389 * Q.sum_pt_top5 - 1.663186
    if Q.sj2_dr < 0.1294903:
        z += 6.654036 * Q.sj2_dr - 0.8616331
    if Q.mean_phi2 < 8.836697e-05:
        z += -21375.56 * Q.mean_phi2 + 1.888894
    if Q.lam1 < 0.00733008 and Q.max_dr < 0.2507612:
        z += 898.4921 * (0.00733008 - Q.lam1) * (0.2507612 - Q.max_dr)
    if Q.sum_pt_top5 < 752.1 and Q.D2_b2 > 0.1830092:
        z += 0.002539198 * (752.1 - Q.sum_pt_top5) * (Q.D2_b2 - 0.1830092)
    if Q.log_sum_pt < 6.605974 and Q.D2_b2 > 0.01618699:
        z += -1.871579 * (6.605974 - Q.log_sum_pt) * (Q.D2_b2 - 0.01618699)
    if Q.tau1 < 0.04369778 and Q.planar_flow < 0.8986489:
        z += -39.07819 * (0.04369778 - Q.tau1) * (0.8986489 - Q.planar_flow)
    if Q.z_7 < 0.0586137 and Q.absphi_0 < 0.07897949:
        z += -510.2641 * (0.0586137 - Q.z_7) * (0.07897949 - Q.absphi_0)
    if Q.log_sum_pt > 6.080494 and Q.mean_phi2 < 8.836697e-05:
        z += -34094.25 * (Q.log_sum_pt - 6.080494) * (8.836697e-05 - Q.mean_phi2)
    if Q.log_sum_pt > 6.896095 and Q.absphi_0 < 0.1056549:
        z += 200.8703 * (Q.log_sum_pt - 6.896095) * (0.1056549 - Q.absphi_0)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.3323192
    if 0.06663269 <= Q.girth < 0.1019409:
        z += 32.12758 * Q.girth - 2.140747
    if 0.1019409 <= Q.girth < 0.1245537:
        z += -108.9102 * Q.girth + 12.23678
    if Q.girth >= 0.1245537:
        z += 7.014214 * Q.girth - 2.202044
    if Q.sj2_dr >= 0.1682655:
        z += 28.9569 * Q.sj2_dr - 4.872449
    if 0.1876504 <= Q.sd_rg < 0.2787955:
        z += -5.307225 * Q.sd_rg + 0.995903
    if Q.sd_rg >= 0.2787955:
        z += -29.63958 * Q.sd_rg + 7.779656
    if Q.tau1 >= 0.1136369:
        z += 22.5753 * Q.tau1 - 2.565387
    if Q.sj3_dr_max >= 0.213399:
        z += 21.13648 * Q.sj3_dr_max - 4.510504
    if Q.LHA >= 0.2931906:
        z += 12.61396 * Q.LHA - 3.698296
    if Q.lam1 >= 0.01200373:
        z += -232.3616 * Q.lam1 + 2.789205
    if Q.lam2 < 0.003408389:
        z += 847.2589 * Q.lam2 - 2.887788
    if Q.sj2_dr > 0.1682655 and Q.log_sum_pt > 6.080494:
        z += -38.48489 * (Q.sj2_dr - 0.1682655) * (Q.log_sum_pt - 6.080494)
    if Q.sj2_dr > 0.1682655 and Q.C2 < 0.09482124:
        z += 514.3259 * (Q.sj2_dr - 0.1682655) * (0.09482124 - Q.C2)
    if Q.sj2_dr > 0.1682655 and Q.n_dr_0_0p05 > 2.0:
        z += -5.490172 * (Q.sj2_dr - 0.1682655) * (Q.n_dr_0_0p05 - 2.0)
    if Q.max_dr > 0.2507612 and Q.z_dr_0_0p05 < 0.6080732:
        z += -83.99915 * (Q.max_dr - 0.2507612) * (0.6080732 - Q.z_dr_0_0p05)
    if Q.girth > 0.1245537 and Q.eccentricity > 0.7792127:
        z += -275.9715 * (Q.girth - 0.1245537) * (Q.eccentricity - 0.7792127)
    if Q.girth > 0.06663269 and Q.eccentricity > 0.7117266:
        z += 143.5252 * (Q.girth - 0.06663269) * (Q.eccentricity - 0.7117266)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.9318167
    if Q.N2 < 0.2233283:
        z += -24.23843 * Q.N2 + 5.413126
    if Q.lam1 < 0.002464291:
        z += -223.7486 * Q.lam1 - 4.503037
    if 0.002464291 <= Q.lam1 < 0.006506576:
        z += 1060.522 * Q.lam1 - 7.667854
    if 0.006506576 <= Q.lam1 < 0.00733008:
        z += 543.9304 * Q.lam1 - 4.306612
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 305.6539 * Q.lam1 - 2.560026
    if Q.e3 < 0.0001869378:
        z += -26896.41 * Q.e3 + 5.027956
    if 0.169029 <= Q.sj3_dr_max < 0.233678:
        z += 10.0132 * Q.sj3_dr_max - 1.692521
    if Q.sj3_dr_max >= 0.233678:
        z += -4.708682 * Q.sj3_dr_max + 1.747658
    if Q.dr_0 < 0.08082334:
        z += -10.47254 * Q.dr_0 + 0.8464254
    if Q.lam2 < 0.000537286:
        z += 7939.4 * Q.lam2 - 5.892114
    if 0.000537286 <= Q.lam2 < 0.001130645:
        z += 2740.982 * Q.lam2 - 3.099076
    if Q.sum_pt < 763.825:
        z += 0.005707467 * Q.sum_pt - 4.359506
    if Q.log_sum_pt >= 6.327379:
        z += 0.8920238 * Q.log_sum_pt - 5.644172
    if Q.C2 >= 0.01867771:
        z += -32.1261 * Q.C2 + 0.6000421
    if Q.tau21 < 0.33555:
        z += 10.18002 * Q.tau21 - 3.415905
    if Q.sj3_dr_min < 0.2089872:
        z += -38.82959 * Q.sj3_dr_min + 8.114889
    if Q.sj3_dr_max > 0.233678 and Q.log_sum_pt > 6.267538:
        z += -111.0714 * (Q.sj3_dr_max - 0.233678) * (Q.log_sum_pt - 6.267538)
    if Q.sum_pt < 763.825 and Q.dr_1 < 0.06108421:
        z += -0.1358502 * (763.825 - Q.sum_pt) * (0.06108421 - Q.dr_1)
    if Q.sj3_dr_max > 0.233678 and Q.sj2_zsoft < 0.05966518:
        z += 1189.989 * (Q.sj3_dr_max - 0.233678) * (0.05966518 - Q.sj2_zsoft)
    if Q.max_dr > 0.121681 and Q.C2_b2 < 0.009032972:
        z += 1796.563 * (Q.max_dr - 0.121681) * (0.009032972 - Q.C2_b2)
    if Q.dr_0 < 0.08082334 and Q.centroid_offset > 0.009480685:
        z += 2146.475 * (0.08082334 - Q.dr_0) * (Q.centroid_offset - 0.009480685)
    if Q.sj3_dr_min < 0.2089872 and Q.lam2 < 0.003408389:
        z += -7290.462 * (0.2089872 - Q.sj3_dr_min) * (0.003408389 - Q.lam2)
    if Q.sj2_dr > 0.2179769 and Q.lam2 > 0.001130645:
        z += -16126.7 * (Q.sj2_dr - 0.2179769) * (Q.lam2 - 0.001130645)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.9763238
    if Q.LHA < 0.2160559:
        z += -10.42107 * Q.LHA + 2.251533
    if Q.lam1 < 7.12483e-05:
        z += -44278.69 * Q.lam1 + 9.321385
    if 7.12483e-05 <= Q.lam1 < 0.0008722282:
        z += -5339.776 * Q.lam1 + 6.547054
    if 0.0008722282 <= Q.lam1 < 0.002464291:
        z += -865.8945 * Q.lam1 + 2.644808
    if 0.002464291 <= Q.lam1 < 0.00483998:
        z += -215.092 * Q.lam1 + 1.041041
    if Q.tau1 < 0.0262518:
        z += 60.53617 * Q.tau1 - 1.589184
    if Q.sum_pt < 739.5:
        z += 0.00930297 * Q.sum_pt - 6.879546
    if Q.centroid_offset < 0.03117077:
        z += 69.77509 * Q.centroid_offset - 2.174944
    if Q.n_pt_above_50 >= 6.0:
        z += -0.9804922 * Q.n_pt_above_50 + 5.882953
    if Q.z_7 < 0.02807091:
        z += -281.267 * Q.z_7 + 7.895423
    if Q.dr_0 < 0.06413297:
        z += 59.82372 * Q.dr_0 - 3.836673
    if Q.e3 < 0.0001869378:
        z += -11734.71 * Q.e3 + 2.193661
    if Q.LHA < 0.2160559 and Q.z_6 < 0.09543973:
        z += -490.571 * (0.2160559 - Q.LHA) * (0.09543973 - Q.z_6)
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -81.64232 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.e2 < 0.03556091 and Q.z_7 < 0.0753896:
        z += 1643.64 * (0.03556091 - Q.e2) * (0.0753896 - Q.z_7)
    if Q.e2 < 0.03556091 and Q.log_sum_pt > 6.896095:
        z += -2025.761 * (0.03556091 - Q.e2) * (Q.log_sum_pt - 6.896095)
    if Q.lam1 < 0.002464291 and Q.centroid_offset < 0.01837778:
        z += 83326.89 * (0.002464291 - Q.lam1) * (0.01837778 - Q.centroid_offset)
    if Q.girth < 0.02054282 and Q.sum_pt_top5 > 716.8828:
        z += 1.771601 * (0.02054282 - Q.girth) * (Q.sum_pt_top5 - 716.8828)
    if Q.log_sum_pt > 6.804164 and Q.e3 < 5.334511e-05:
        z += -526153.6 * (Q.log_sum_pt - 6.804164) * (5.334511e-05 - Q.e3)
    if Q.sum_pt < 739.5 and Q.mean_eta2 < 0.008675069:
        z += 0.5918074 * (739.5 - Q.sum_pt) * (0.008675069 - Q.mean_eta2)
    if Q.lam1 < 7.12483e-05 and Q.pt_5 < 73.75:
        z += -1091.212 * (7.12483e-05 - Q.lam1) * (73.75 - Q.pt_5)
    return max(0.0, z)


def neuron_6(Q):
    z = 1.025924
    if Q.log_sum_pt < 6.423044:
        z += 9.103863 * Q.log_sum_pt - 58.47451
    if 0.1591713 <= Q.sj2_dr < 0.1682655:
        z += -16.39722 * Q.sj2_dr + 2.609967
    if Q.sj2_dr >= 0.1682655:
        z += 3.709175 * Q.sj2_dr - 0.7732468
    if Q.centroid_offset < 0.01837778:
        z += 91.71122 * Q.centroid_offset - 1.685449
    if 0.08723651 <= Q.girth < 0.1019409:
        z += -217.2567 * Q.girth + 18.95272
    if 0.1019409 <= Q.girth < 0.1484084:
        z += -376.7796 * Q.girth + 35.21463
    if Q.girth >= 0.1484084:
        z += -254.1975 * Q.girth + 17.02242
    if Q.pt_6 < 31.90625:
        z += 0.4394124 * Q.pt_6 - 16.65313
    if 31.90625 <= Q.pt_6 < 39.75:
        z += 0.3356975 * Q.pt_6 - 13.34397
    if Q.lam2 < 0.001130645:
        z += 3098.688 * Q.lam2 - 3.503515
    if 0.251526 <= Q.LHA < 0.3255822:
        z += 46.6815 * Q.LHA - 11.74161
    if Q.LHA >= 0.3255822:
        z += 97.95004 * Q.LHA - 28.43374
    if Q.C2 < 0.03578649:
        z += -77.27497 * Q.C2 + 2.7654
    if 0.03464708 <= Q.sj3_dr_max < 0.1789613:
        z += -15.35558 * Q.sj3_dr_max + 0.5320261
    if Q.sj3_dr_max >= 0.1789613:
        z += 2.115994 * Q.sj3_dr_max - 2.594711
    if Q.z_6 < 0.04355037:
        z += -355.2149 * Q.z_6 + 15.46974
    if Q.max_dr < 0.177305:
        z += 36.36003 * Q.max_dr - 6.446815
    if Q.log_sum_pt < 6.423044 and Q.pt_7 < 38.53125:
        z += 1.534016 * (6.423044 - Q.log_sum_pt) * (38.53125 - Q.pt_7)
    if Q.centroid_offset > 0.00809236 and Q.N2 > 0.198361:
        z += 395.6251 * (Q.centroid_offset - 0.00809236) * (Q.N2 - 0.198361)
    if Q.sj2_dr > 0.1591713 and Q.sj2_zsoft > 0.05966518:
        z += 154.9852 * (Q.sj2_dr - 0.1591713) * (Q.sj2_zsoft - 0.05966518)
    if Q.centroid_offset < 0.01837778 and Q.planar_flow < 0.1115136:
        z += -890.0414 * (0.01837778 - Q.centroid_offset) * (0.1115136 - Q.planar_flow)
    if Q.pt_6 < 39.75 and Q.sum_pt_top3 < 711.875:
        z += 0.001849816 * (39.75 - Q.pt_6) * (711.875 - Q.sum_pt_top3)
    if Q.centroid_offset > 0.00809236 and Q.sj3_dr_min < 0.2089872:
        z += 584.8752 * (Q.centroid_offset - 0.00809236) * (0.2089872 - Q.sj3_dr_min)
    if Q.centroid_offset > 0.00809236 and Q.mean_phi2 < 0.01426135:
        z += 4115.471 * (Q.centroid_offset - 0.00809236) * (0.01426135 - Q.mean_phi2)
    if Q.centroid_offset > 0.02076709 and Q.n_dr_0_0p05 < 5.0:
        z += -11.39641 * (Q.centroid_offset - 0.02076709) * (5.0 - Q.n_dr_0_0p05)
    if Q.sj3_dr_max > 0.1789613 and Q.D2_b2 < 1.59265:
        z += 23.39769 * (Q.sj3_dr_max - 0.1789613) * (1.59265 - Q.D2_b2)
    if Q.pt_6 < 39.75 and Q.z_top3_slots < 0.7861046:
        z += -2.074413 * (39.75 - Q.pt_6) * (0.7861046 - Q.z_top3_slots)
    if Q.lam2 < 0.003408389 and Q.n_dr_0p1_0p2 > 1.0:
        z += 90.94 * (0.003408389 - Q.lam2) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.centroid_offset < 0.01837778 and Q.n_dr_0p05_0p1 > 0.0:
        z += 25.82862 * (0.01837778 - Q.centroid_offset) * (Q.n_dr_0p05_0p1 - 0.0)
    if Q.girth > 0.08723651 and Q.D2_b2 < 0.380911:
        z += -161.7195 * (Q.girth - 0.08723651) * (0.380911 - Q.D2_b2)
    return max(0.0, z)


def neuron_7(Q):
    z = 1.382631
    if Q.tau1 < 0.06345984:
        z += -70.86866 * Q.tau1 + 6.282768
    if 0.06345984 <= Q.tau1 < 0.08865369:
        z += -36.27938 * Q.tau1 + 4.087738
    if 0.08865369 <= Q.tau1 < 0.1136369:
        z += 34.58928 * Q.tau1 - 2.19503
    if Q.tau1 >= 0.1136369:
        z += 21.38179 * Q.tau1 - 0.6941721
    if 0.0479157 <= Q.girth < 0.1019409:
        z += -65.80942 * Q.girth + 3.153304
    if 0.1019409 <= Q.girth < 0.1245537:
        z += 237.8746 * Q.girth - 27.80453
    if Q.girth >= 0.1245537:
        z += 396.3103 * Q.girth - 47.53828
    if Q.z_dr_0p1_0p2 < 0.1585582:
        z += -5.763358 * Q.z_dr_0p1_0p2 + 0.9138276
    if Q.lam1 < 0.003377388:
        z += 1962.44 * Q.lam1 - 7.837992
    if 0.003377388 <= Q.lam1 < 0.00483998:
        z += 827.347 * Q.lam1 - 4.004343
    if 0.00733008 <= Q.lam1 < 0.01200373:
        z += 551.0126 * Q.lam1 - 4.038967
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += 1275.012 * Q.lam1 - 12.72966
    if Q.lam1 >= 0.01643375:
        z += -1612.827 * Q.lam1 + 34.72836
    if Q.LHA < 0.1329373:
        z += -12.89825 * Q.LHA + 3.026168
    if 0.1329373 <= Q.LHA < 0.2346184:
        z += 1.223347 * Q.LHA + 1.148881
    if 0.2346184 <= Q.LHA < 0.3467135:
        z += 14.1216 * Q.LHA - 1.877287
    if Q.LHA >= 0.3467135:
        z += -272.9301 * Q.LHA + 97.6474
    if Q.girth2_top5 < 0.0006570502:
        z += 1815.517 * Q.girth2_top5 - 1.192886
    if Q.centroid_offset < 0.02685622:
        z += -116.1231 * Q.centroid_offset + 3.118627
    if Q.sj3_dr_max < 0.1594012:
        z += -19.57606 * Q.sj3_dr_max + 2.208801
    if 0.1594012 <= Q.sj3_dr_max < 0.213399:
        z += 16.883 * Q.sj3_dr_max - 3.602814
    if Q.e3 < 8.147744e-05:
        z += 31455.02 * Q.e3 - 2.562875
    if Q.planar_flow < 0.1950135 and Q.lam1 > 0.008375572:
        z += -5965.066 * (0.1950135 - Q.planar_flow) * (Q.lam1 - 0.008375572)
    if Q.z_dr_0p1_0p2 < 0.1585582 and Q.centroid_offset < 0.02685622:
        z += -464.876 * (0.1585582 - Q.z_dr_0p1_0p2) * (0.02685622 - Q.centroid_offset)
    if Q.planar_flow < 0.1950135 and Q.log_sum_pt > 6.46415:
        z += 29.2791 * (0.1950135 - Q.planar_flow) * (Q.log_sum_pt - 6.46415)
    if Q.lam1 > 0.002464291 and Q.sj3_dr_max < 0.1879486:
        z += -11632.84 * (Q.lam1 - 0.002464291) * (0.1879486 - Q.sj3_dr_max)
    if Q.lam1 > 0.00733008 and Q.D2 < 2.843757:
        z += -366.9554 * (Q.lam1 - 0.00733008) * (2.843757 - Q.D2)
    if Q.lam1 > 0.002464291 and Q.planar_flow < 0.1950135:
        z += 626.6568 * (Q.lam1 - 0.002464291) * (0.1950135 - Q.planar_flow)
    if Q.lam1 > 0.002464291 and Q.D2 > 0.2568137:
        z += -321.1186 * (Q.lam1 - 0.002464291) * (Q.D2 - 0.2568137)
    if Q.lam1 > 0.01643375 and Q.D2 < 3.885568:
        z += 653.6335 * (Q.lam1 - 0.01643375) * (3.885568 - Q.D2)
    if Q.girth > 0.0479157 and Q.D2 < 2.357246:
        z += 63.98458 * (Q.girth - 0.0479157) * (2.357246 - Q.D2)
    if Q.e3 < 8.147744e-05 and Q.pt_7 < 43.5:
        z += -1184.834 * (8.147744e-05 - Q.e3) * (43.5 - Q.pt_7)
    if Q.girth > 0.08065885 and Q.D2 < 2.843757:
        z += -32.64482 * (Q.girth - 0.08065885) * (2.843757 - Q.D2)
    if Q.centroid_offset < 0.02685622 and Q.n_dr_0p2_0p4 < 1.0:
        z += -47.24057 * (0.02685622 - Q.centroid_offset) * (1.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_8(Q):
    z = -1.420138
    if Q.tau1 < 0.019014:
        z += 128.7385 * Q.tau1 - 2.447835
    if Q.LHA < 0.1767241:
        z += -25.98419 * Q.LHA + 4.592033
    if Q.log_sum_pt >= 6.701242:
        z += -12.44077 * Q.log_sum_pt + 83.36862
    if Q.z_dr_0_0p05 >= 0.9008535:
        z += -8.603189 * Q.z_dr_0_0p05 + 7.750213
    if Q.sj3_dr_max < 0.1070199:
        z += 88.83074 * Q.sj3_dr_max - 8.321596
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += -25.52559 * Q.sj3_dr_max + 3.916812
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += -4.935998 * Q.sj3_dr_max + 0.9804235
    if Q.girth < 0.03360421:
        z += 90.44005 * Q.girth - 3.039166
    if Q.lam2 < 9.651826e-05:
        z += -18689.18 * Q.lam2 + 1.803847
    if Q.centroid_offset >= 0.006789738:
        z += -100.5871 * Q.centroid_offset + 0.6829598
    if Q.lam1 < 0.00733008 and Q.centroid_offset < 0.02355416:
        z += 22894.09 * (0.00733008 - Q.lam1) * (0.02355416 - Q.centroid_offset)
    if Q.LHA < 0.251526 and Q.z_7 > 0.01685855:
        z += -463.9392 * (0.251526 - Q.LHA) * (Q.z_7 - 0.01685855)
    if Q.sj3_dr_max < 0.1986272 and Q.n_dr_0p05_0p1 < 5.0:
        z += 2.764438 * (0.1986272 - Q.sj3_dr_max) * (5.0 - Q.n_dr_0p05_0p1)
    if Q.LHA < 0.1767241 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -1272.463 * (0.1767241 - Q.LHA) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.lam1 < 0.005433361 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += 8226.592 * (0.005433361 - Q.lam1) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.tau1 < 0.0262518 and Q.centroid_offset < 0.03117077:
        z += 9685.26 * (0.0262518 - Q.tau1) * (0.03117077 - Q.centroid_offset)
    if Q.girth < 0.03360421 and Q.pt_7 > 15.55391:
        z += 3.681667 * (0.03360421 - Q.girth) * (Q.pt_7 - 15.55391)
    if Q.girth < 0.03360421 and Q.centroid_offset < 0.03117077:
        z += -2912.437 * (0.03360421 - Q.girth) * (0.03117077 - Q.centroid_offset)
    if Q.lam1 < 0.005433361 and Q.centroid_offset < 0.03117077:
        z += 10389.55 * (0.005433361 - Q.lam1) * (0.03117077 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.1070199 and Q.centroid_offset > 0.01627885:
        z += -12883.79 * (0.1070199 - Q.sj3_dr_max) * (Q.centroid_offset - 0.01627885)
    if Q.sj3_dr_max < 0.1594012 and Q.centroid_offset > 0.003343241:
        z += 1174.326 * (0.1594012 - Q.sj3_dr_max) * (Q.centroid_offset - 0.003343241)
    if Q.sj3_dr_max < 0.1070199 and Q.girth2_top3 < 0.0153634:
        z += 8386.412 * (0.1070199 - Q.sj3_dr_max) * (0.0153634 - Q.girth2_top3)
    if Q.sj3_dr_max < 0.1426152 and Q.girth2_top3 < 0.0153634:
        z += -4363.721 * (0.1426152 - Q.sj3_dr_max) * (0.0153634 - Q.girth2_top3)
    return max(0.0, z)


def neuron_9(Q):
    z = -6.256422
    if Q.girth < 0.02689598:
        z += 36.30046 * Q.girth - 3.251822
    if 0.02689598 <= Q.girth < 0.05464922:
        z += 81.9899 * Q.girth - 4.480684
    if Q.sum_pt < 813.4156:
        z += -0.011811 * Q.sum_pt + 10.61472
    if 813.4156 <= Q.sum_pt < 988.4078:
        z += -0.005757236 * Q.sum_pt + 5.690497
    if Q.lam2 < 0.0001947983:
        z += -9250.984 * Q.lam2 + 1.802076
    if Q.lam2 >= 0.000537286:
        z += 902.421 * Q.lam2 - 0.4848582
    if Q.lam1 < 0.003377388:
        z += -817.0883 * Q.lam1 + 2.759624
    if Q.sj2_dr < 0.1294903:
        z += 10.95407 * Q.sj2_dr - 1.418445
    if Q.sj3_dr_min >= 0.1278212:
        z += 21.61759 * Q.sj3_dr_min - 2.763186
    if Q.sj3_dr_max < 0.1426152:
        z += 18.99748 * Q.sj3_dr_max - 2.317982
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += -5.528773 * Q.sj3_dr_max + 1.179835
    if Q.e3 < 1.960701e-05:
        z += 57183.82 * Q.e3 - 1.121204
    if Q.max_dr < 0.1027585:
        z += -13.10754 * Q.max_dr + 1.346912
    if Q.centroid_offset < 0.03117077:
        z += -110.4794 * Q.centroid_offset + 3.443727
    if Q.lam1 < 0.00595415 and Q.lam2 < 0.001130645:
        z += 1206957.0 * (0.00595415 - Q.lam1) * (0.001130645 - Q.lam2)
    if Q.lam1 < 0.00595415 and Q.sj3_dr_min < 0.08543881:
        z += -3588.354 * (0.00595415 - Q.lam1) * (0.08543881 - Q.sj3_dr_min)
    if Q.centroid_offset < 0.02355416 and Q.lam1 < 0.00733008:
        z += 45070.98 * (0.02355416 - Q.centroid_offset) * (0.00733008 - Q.lam1)
    if Q.lam2 > 0.000537286 and Q.eccentricity < 0.9979797:
        z += -1020.601 * (Q.lam2 - 0.000537286) * (0.9979797 - Q.eccentricity)
    return max(0.0, z)


def neuron_10(Q):
    z = 3.751985
    z += -29.32363 * Q.e2
    if Q.sj3_dr_min >= 0.1278212:
        z += 18.14869 * Q.sj3_dr_min - 2.319786
    if Q.lam1 < 0.003377388:
        z += 1197.074 * Q.lam1 - 2.410807
    if 0.003377388 <= Q.lam1 < 0.00483998:
        z += 483.2658 * Q.lam1
    if Q.lam1 >= 0.00483998:
        z += 292.5669 * Q.lam1 + 0.9229789
    if 0.0001947983 <= Q.lam2 < 0.003408389:
        z += 1221.532 * Q.lam2 - 0.2379524
    if Q.lam2 >= 0.003408389:
        z += 268.3797 * Q.lam2 + 3.010762
    if Q.e3 >= 8.147744e-05:
        z += 1669.895 * Q.e3 - 0.1360588
    if Q.tau1 >= 0.04369778:
        z += 13.18152 * Q.tau1 - 0.5760031
    if Q.girth2_top3 < 0.002915531:
        z += -353.0194 * Q.girth2_top3 + 1.029239
    if Q.LHA >= 0.1967397:
        z += -13.70652 * Q.LHA + 2.696617
    if Q.girth < 0.1484084:
        z += 24.0279 * Q.girth - 3.565943
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += 20.44237 * Q.sj2_dr - 2.647088
    if Q.sj2_dr >= 0.1591713:
        z += -3.173412 * Q.sj2_dr + 1.111866
    if Q.lam2 > 0.0001947983 and Q.eccentricity > 0.4829602:
        z += -2276.758 * (Q.lam2 - 0.0001947983) * (Q.eccentricity - 0.4829602)
    if Q.lam2 > 0.0001947983 and Q.planar_flow > 0.06126205:
        z += -580.8045 * (Q.lam2 - 0.0001947983) * (Q.planar_flow - 0.06126205)
    if Q.sj2_dr > 0.1682655 and Q.planar_flow < 0.6947818:
        z += -32.48497 * (Q.sj2_dr - 0.1682655) * (0.6947818 - Q.planar_flow)
    if Q.LHA > 0.3127275 and Q.planar_flow < 0.6947818:
        z += -45.76576 * (Q.LHA - 0.3127275) * (0.6947818 - Q.planar_flow)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.5345149
    if Q.planar_flow < 0.2534037:
        z += -11.31431 * Q.planar_flow + 2.867089
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += 32.4556 * Q.sj2_dr - 4.202686
    if 0.1591713 <= Q.sj2_dr < 0.1778793:
        z += 13.97338 * Q.sj2_dr - 1.260847
    if Q.sj2_dr >= 0.1778793:
        z += 26.76591 * Q.sj2_dr - 3.536373
    if Q.pt_7 < 45.75:
        z += 0.05684124 * Q.pt_7 - 2.600487
    if Q.lam1 < 0.008375572:
        z += -572.092 * Q.lam1 + 5.742898
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += -398.615 * Q.lam1 + 4.28993
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += 111.7232 * Q.lam1 - 1.836031
    if Q.girth < 0.0717028:
        z += 58.80304 * Q.girth - 4.216343
    if Q.centroid_offset < 0.01837778:
        z += -157.0387 * Q.centroid_offset + 2.886023
    if 0.1789613 <= Q.sj3_dr_max < 0.2623172:
        z += -49.57128 * Q.sj3_dr_max + 8.871342
    if Q.sj3_dr_max >= 0.2623172:
        z += -40.00105 * Q.sj3_dr_max + 6.360907
    if Q.planar_flow < 0.2534037 and Q.centroid_offset < 0.04990367:
        z += -201.2931 * (0.2534037 - Q.planar_flow) * (0.04990367 - Q.centroid_offset)
    if Q.sj2_dr > 0.2414351 and Q.sum_pt < 739.5:
        z += 0.03272677 * (Q.sj2_dr - 0.2414351) * (739.5 - Q.sum_pt)
    if Q.lam1 < 0.003377388 and Q.centroid_offset < 0.02685622:
        z += -60326.8 * (0.003377388 - Q.lam1) * (0.02685622 - Q.centroid_offset)
    if Q.lam1 < 0.01200373 and Q.centroid_offset > 0.01627885:
        z += -18007.12 * (0.01200373 - Q.lam1) * (Q.centroid_offset - 0.01627885)
    if Q.lam1 < 0.01643375 and Q.lam2 < 0.001130645:
        z += 111239.3 * (0.01643375 - Q.lam1) * (0.001130645 - Q.lam2)
    return max(0.0, z)


def neuron_12(Q):
    z = -1.765716
    if Q.girth >= 0.1245537:
        z += 132.1238 * Q.girth - 16.45651
    if Q.sd_rg >= 0.324646:
        z += 87.68851 * Q.sd_rg - 28.46773
    if Q.e2 >= 0.06344108:
        z += -72.76793 * Q.e2 + 4.616476
    if Q.sd_rg > 0.324646 and Q.log_sum_pt < 6.896095:
        z += -239.8497 * (Q.sd_rg - 0.324646) * (6.896095 - Q.log_sum_pt)
    if Q.girth > 0.1245537 and Q.pt_6 < 62.25:
        z += -1.193116 * (Q.girth - 0.1245537) * (62.25 - Q.pt_6)
    if Q.sd_rg > 0.324646 and Q.log_sum_pt < 6.701242:
        z += 181.0457 * (Q.sd_rg - 0.324646) * (6.701242 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.8153515
    if Q.girth < 0.1484084:
        z += -76.04103 * Q.girth + 11.28513
    if Q.lam1 < 0.01643375:
        z += -264.8828 * Q.lam1 + 4.353018
    if Q.sj3_dr_max >= 0.233678:
        z += -11.90903 * Q.sj3_dr_max + 2.782879
    if Q.centroid_offset < 0.01837778:
        z += 88.48718 * Q.centroid_offset - 1.626198
    if Q.sj3_dr_min < 0.1278212:
        z += 20.60703 * Q.sj3_dr_min - 2.634015
    if Q.pt_7 >= 31.85938:
        z += -0.08577209 * Q.pt_7 + 2.732645
    if Q.tau1 < 0.1136369:
        z += 13.57129 * Q.tau1 - 1.5422
    if Q.z_7 >= 0.04624032:
        z += 57.09915 * Q.z_7 - 2.640283
    if Q.log_sum_pt >= 6.670067:
        z += -3.748745 * Q.log_sum_pt + 25.00438
    if Q.girth < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -99.48348 * (0.1484084 - Q.girth) * (6.804164 - Q.log_sum_pt)
    if Q.girth < 0.1484084 and Q.pt_7 < 38.53125:
        z += -0.5054877 * (0.1484084 - Q.girth) * (38.53125 - Q.pt_7)
    if Q.girth < 0.1484084 and Q.tau2 < 0.03585464:
        z += -633.0148 * (0.1484084 - Q.girth) * (0.03585464 - Q.tau2)
    if Q.sum_pt_top5 > 687.4375 and Q.pt_7 < 40.04062:
        z += 0.0006952303 * (Q.sum_pt_top5 - 687.4375) * (40.04062 - Q.pt_7)
    if Q.lam1 < 0.01643375 and Q.pt_7 < 25.57812:
        z += -18.26279 * (0.01643375 - Q.lam1) * (25.57812 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = -4.342875
    if Q.lam2 < 0.003408389:
        z += 2053.662 * Q.lam2 - 6.999679
    if Q.sd_rg < 0.1449048:
        z += -11.12633 * Q.sd_rg + 2.593458
    if 0.1449048 <= Q.sd_rg < 0.1572969:
        z += -42.65766 * Q.sd_rg + 7.162498
    if 0.1572969 <= Q.sd_rg < 0.1876504:
        z += 17.23481 * Q.sd_rg - 2.2584
    if 0.1876504 <= Q.sd_rg < 0.2330919:
        z += -42.36932 * Q.sd_rg + 8.926339
    if Q.sd_rg >= 0.2330919:
        z += -31.24299 * Q.sd_rg + 6.332882
    if Q.lam1 < 0.006506576:
        z += 1154.06 * Q.lam1 - 6.493915
    if 0.006506576 <= Q.lam1 < 0.01200373:
        z += -184.6529 * Q.lam1 + 2.216523
    if 0.02689598 <= Q.girth < 0.06663269:
        z += 102.5585 * Q.girth - 2.758411
    if 0.06663269 <= Q.girth < 0.08065885:
        z += 153.3854 * Q.girth - 6.145144
    if 0.08065885 <= Q.girth < 0.08723651:
        z += 243.19 * Q.girth - 13.38868
    if Q.girth >= 0.08723651:
        z += -102.0166 * Q.girth + 16.72594
    if Q.z_dr_0p05_0p1 >= 0.163898:
        z += -1.923211 * Q.z_dr_0p05_0p1 + 0.3152106
    if Q.centroid_offset >= 0.04990367:
        z += -1194.081 * Q.centroid_offset + 59.589
    if Q.tau1 < 0.09538712:
        z += -101.5262 * Q.tau1 + 11.00043
    if 0.09538712 <= Q.tau1 < 0.1416226:
        z += -28.46598 * Q.tau1 + 4.031427
    if Q.LHA >= 0.3033137:
        z += -32.19726 * Q.LHA + 9.765872
    if Q.e3 < 8.147744e-05:
        z += 15914.9 * Q.e3 - 1.296705
    if Q.e2 < 0.04110972:
        z += -108.9114 * Q.e2 + 4.477318
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -18236.17 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.01643375:
        z += 1539.565 * (0.1115136 - Q.planar_flow) * (0.01643375 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00595415:
        z += 12551.75 * (0.1115136 - Q.planar_flow) * (0.00595415 - Q.lam1)
    if Q.lam2 < 0.003408389 and Q.sd_rg > 0.2037854:
        z += -5311.186 * (0.003408389 - Q.lam2) * (Q.sd_rg - 0.2037854)
    if Q.planar_flow < 0.1115136 and Q.sj2_zsoft < 0.2832687:
        z += 89.60398 * (0.1115136 - Q.planar_flow) * (0.2832687 - Q.sj2_zsoft)
    if Q.lam2 < 0.003408389 and Q.LHA > 0.1767241:
        z += 6257.921 * (0.003408389 - Q.lam2) * (Q.LHA - 0.1767241)
    if Q.lam2 < 0.003408389 and Q.sd_rg > 0.1572969:
        z += 14442.24 * (0.003408389 - Q.lam2) * (Q.sd_rg - 0.1572969)
    if Q.z_dr_0p05_0p1 > 0.163898 and Q.n_dr_0p2_0p4 < 1.0:
        z += 5.476425 * (Q.z_dr_0p05_0p1 - 0.163898) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.z_dr_0p05_0p1 > 0.163898 and Q.log_sum_pt < 6.670067:
        z += -4.98329 * (Q.z_dr_0p05_0p1 - 0.163898) * (6.670067 - Q.log_sum_pt)
    if Q.lam1 < 0.01643375 and Q.centroid_offset > 0.02685622:
        z += -19137.33 * (0.01643375 - Q.lam1) * (Q.centroid_offset - 0.02685622)
    if Q.z_dr_0p05_0p1 > 0.163898 and Q.z_dr_0p2_0p4 < 0.20552:
        z += -9.384049 * (Q.z_dr_0p05_0p1 - 0.163898) * (0.20552 - Q.z_dr_0p2_0p4)
    return max(0.0, z)


def neuron_15(Q):
    z = -1.860103
    if Q.lam2 < 0.003408389:
        z += -930.9247 * Q.lam2 + 3.172954
    if Q.lam1 < 0.00595415:
        z += 727.3127 * Q.lam1 - 4.559547
    if 0.00595415 <= Q.lam1 < 0.006506576:
        z += 1264.102 * Q.lam1 - 7.755673
    if 0.006506576 <= Q.lam1 < 0.00733008:
        z += 636.0176 * Q.lam1 - 3.668993
    if 0.00733008 <= Q.lam1 < 0.01643375:
        z += -109.0843 * Q.lam1 + 1.792663
    if Q.tau1 < 0.05356915:
        z += -128.1581 * Q.tau1 + 6.865323
    if 0.08865369 <= Q.tau1 < 0.09538712:
        z += -81.19156 * Q.tau1 + 7.197931
    if 0.09538712 <= Q.tau1 < 0.1027642:
        z += -112.33 * Q.tau1 + 10.16813
    if 0.1027642 <= Q.tau1 < 0.1136369:
        z += 22.77314 * Q.tau1 - 3.715633
    if 0.1136369 <= Q.tau1 < 0.1416226:
        z += 101.9475 * Q.tau1 - 12.71276
    if Q.tau1 >= 0.1416226:
        z += -273.5291 * Q.tau1 + 40.46323
    if Q.girth2_top3 < 0.002151568:
        z += -955.7753 * Q.girth2_top3 + 2.056415
    if Q.girth2_top2 < 0.007639643:
        z += -184.1904 * Q.girth2_top2 + 1.407149
    if Q.eccentricity >= 0.9458207:
        z += -31.59606 * Q.eccentricity + 29.88421
    if Q.girth < 0.03360421:
        z += 280.6407 * Q.girth - 14.18513
    if 0.03360421 <= Q.girth < 0.0717028:
        z += 124.7926 * Q.girth - 8.947977
    if Q.N2 < 0.2233283 and Q.sj2_dr < 0.1872617:
        z += -346.7687 * (0.2233283 - Q.N2) * (0.1872617 - Q.sj2_dr)
    if Q.N2 < 0.2233283 and Q.lam1 > 0.008375572:
        z += -4362.054 * (0.2233283 - Q.N2) * (Q.lam1 - 0.008375572)
    if Q.N2 < 0.2233283 and Q.LHA > 0.2931906:
        z += 569.3721 * (0.2233283 - Q.N2) * (Q.LHA - 0.2931906)
    if Q.N2 < 0.2233283 and Q.LHA > 0.4235925:
        z += -867.9489 * (0.2233283 - Q.N2) * (Q.LHA - 0.4235925)
    if Q.N2 < 0.2233283 and Q.LHA > 0.3255822:
        z += -653.6724 * (0.2233283 - Q.N2) * (Q.LHA - 0.3255822)
    if Q.N2 < 0.2233283 and Q.LHA > 0.3855647:
        z += -2664.486 * (0.2233283 - Q.N2) * (Q.LHA - 0.3855647)
    if Q.girth2_top3 < 0.002151568 and Q.planar_flow < 0.1950135:
        z += -8094.884 * (0.002151568 - Q.girth2_top3) * (0.1950135 - Q.planar_flow)
    if Q.eccentricity > 0.9458207 and Q.sum_pt_top5 > 367.5938:
        z += 0.1516185 * (Q.eccentricity - 0.9458207) * (Q.sum_pt_top5 - 367.5938)
    if Q.lam1 < 0.008375572 and Q.D2 < 0.875672:
        z += -2262.949 * (0.008375572 - Q.lam1) * (0.875672 - Q.D2)
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += 4173.027 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    return [B[c] + sum(h[j] * W[j][c] for j in range(len(h))) for c in range(5)]


def classify(pt, eta, phi):
    s = logits(jet_layer_4(quantities(pt, eta, phi)))
    m = max(s)
    e = [math.exp(x - m) for x in s]
    p = [x / sum(e) for x in e]
    return CLASSES[s.index(m)], s, p


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3][:8] + [0.0] * 0
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4][:8] + [0.0] * 0
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36][:8] + [0.0] * 0
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
