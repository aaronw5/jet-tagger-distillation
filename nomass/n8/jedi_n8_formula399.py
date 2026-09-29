"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; no mass observables), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.3% (the network: 65.8%); same class as the network for 86.7% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
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
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        max_dr=max(dr[i] for i in real),
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_6=z[6],
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_dr_min=min(subjets(3)["dr"]),
        sj2_dr=subjets(2)["dr"][0],
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -13.3
    if Q.C2_b2 < 0.00113:
        z += 718.0 * Q.C2_b2 - 0.81134
    if Q.LHA < 0.24:
        z += 15.9 * Q.LHA - 3.816
    if Q.LHA >= 0.253:
        z += 20.3 * Q.LHA - 5.1359
    if Q.e2_sq < 0.00729:
        z += 1270.0 * Q.e2_sq - 9.3218
    if 0.00729 <= Q.e2_sq < 0.00734:
        z += 2023.0 * Q.e2_sq - 14.81117
    if Q.e2_sq >= 0.00734:
        z += 753.0 * Q.e2_sq - 5.48937
    if Q.girth2 < 0.00569:
        z += -1237.0 * Q.girth2 + 27.08911
    if 0.00569 <= Q.girth2 < 0.0129:
        z += -1698.0 * Q.girth2 + 29.7122
    if 0.0129 <= Q.girth2 < 0.0184:
        z += -1280.0 * Q.girth2 + 24.32
    if 0.0184 <= Q.girth2 < 0.019:
        z += -2400.0 * Q.girth2 + 44.928
    if Q.girth2 >= 0.019:
        z += -1120.0 * Q.girth2 + 20.608
    if Q.girth2_top3 < 0.0154:
        z += 96.3 * Q.girth2_top3 - 1.48302
    if Q.lam1 >= 0.00872:
        z += 272.0 * Q.lam1 - 2.37184
    if Q.lam2 < 0.00131:
        z += -814.0 * Q.lam2 + 1.06634
    if Q.log_sum_pt < 6.71:
        z += 6.72 * Q.log_sum_pt - 45.0912
    if Q.max_dr < 0.0435:
        z += -141.0 * Q.max_dr + 6.1335
    if Q.pt_7 >= 20.3:
        z += -0.125 * Q.pt_7 + 2.5375
    if Q.sj2_dr < 0.219:
        z += 6.83 * Q.sj2_dr - 1.49577
    if Q.sj3_dr_max < 0.175:
        z += 16.7 * Q.sj3_dr_max - 2.9225
    if Q.z_7 < 0.0234:
        z += -27.7 * Q.z_7 - 0.60317
    if 0.0234 <= Q.z_7 < 0.0362:
        z += 86.3 * Q.z_7 - 3.27077
    if 0.0362 <= Q.z_7 < 0.0379:
        z += 173.1 * Q.z_7 - 6.41293
    if Q.z_7 >= 0.0379:
        z += 86.8 * Q.z_7 - 3.14216
    if Q.planar_flow < 0.117 and Q.D2_b2 < 1.26:
        z += 7.77 * (0.117 - Q.planar_flow) * (1.26 - Q.D2_b2)
    if Q.sum_pt < 714.0 and Q.D2_b2 < 1.07:
        z += -0.00599 * (714.0 - Q.sum_pt) * (1.07 - Q.D2_b2)
    return max(0.0, z)


def neuron_1(Q):
    z = -0.772
    if Q.centroid_offset < 0.0126:
        z += 19.3 * Q.centroid_offset - 0.24318
    if Q.e3 >= 5.9e-05:
        z += -464.0 * Q.e3 + 0.027376
    if Q.girth < 0.0723:
        z += -7.2 * Q.girth - 0.40932
    if 0.0723 <= Q.girth < 0.101:
        z += 32.4 * Q.girth - 3.2724
    if Q.girth2 < 0.00722:
        z += 379.0 * Q.girth2 - 2.73638
    if Q.lam1 < 0.00037:
        z += 2830.0 * Q.lam1 - 1.0471
    if Q.log_sum_pt < 6.56:
        z += -5.67 * Q.log_sum_pt + 37.1952
    if Q.max_dr < 0.184:
        z += 4.64 * Q.max_dr - 0.85376
    z += 0.0211 * Q.pt_5
    if 35.0 <= Q.pt_7 < 41.7:
        z += 0.0489 * Q.pt_7 - 1.7115
    if 41.7 <= Q.pt_7 < 53.4:
        z += 0.121 * Q.pt_7 - 4.71807
    if Q.pt_7 >= 53.4:
        z += 0.01 * Q.pt_7 + 1.20933
    if Q.sj2_dr < 0.164:
        z += -10.3 * Q.sj2_dr + 1.6892
    if Q.sum_pt >= 626.0:
        z += 0.00623 * Q.sum_pt - 3.89998
    if Q.sum_pt_top5 < 474.0:
        z += 0.01488 * Q.sum_pt_top5 - 7.60524
    if 474.0 <= Q.sum_pt_top5 < 560.0:
        z += 0.00642 * Q.sum_pt_top5 - 3.5952
    if Q.tau1 < 0.101:
        z += -15.8 * Q.tau1 + 1.5958
    z += 12.3 * Q.z_6
    if Q.z_7 < 0.0663:
        z += 46.1 * Q.z_7 - 3.05643
    if Q.lam1 < 0.00932 and Q.centroid_offset > 0.0117:
        z += 9740.0 * (0.00932 - Q.lam1) * (Q.centroid_offset - 0.0117)
    if Q.lam1 < 0.0096 and Q.pt_5 > 23.4:
        z += -4.85 * (0.0096 - Q.lam1) * (Q.pt_5 - 23.4)
    if Q.lam2 < 0.00317 and Q.tau21_b2 < 0.279:
        z += 706.0 * (0.00317 - Q.lam2) * (0.279 - Q.tau21_b2)
    if Q.log_sum_pt > 6.62 and Q.girth2_top3 < 0.00763:
        z += -558.0 * (Q.log_sum_pt - 6.62) * (0.00763 - Q.girth2_top3)
    if Q.log_sum_pt > 6.6 and Q.tau21_b2 < 0.0292:
        z += 127.0 * (Q.log_sum_pt - 6.6) * (0.0292 - Q.tau21_b2)
    if Q.log_sum_pt > 6.37 and Q.z_dr_0_0p05 < 0.927:
        z += 4.43 * (Q.log_sum_pt - 6.37) * (0.927 - Q.z_dr_0_0p05)
    if Q.pt_7 > 33.3 and Q.e3 < 4.05e-05:
        z += -1400.0 * (Q.pt_7 - 33.3) * (4.05e-05 - Q.e3)
    if Q.width < 0.0106 and Q.planar_flow < 0.251:
        z += -309.0 * (0.0106 - Q.width) * (0.251 - Q.planar_flow)
    if Q.z_7 < 0.0467 and Q.D2_b2 < 0.3:
        z += -129.0 * (0.0467 - Q.z_7) * (0.3 - Q.D2_b2)
    if Q.z_7 < 0.066 and Q.girth2_top2 < 0.0158:
        z += 3050.0 * (0.066 - Q.z_7) * (0.0158 - Q.girth2_top2)
    if Q.z_7 < 0.0573 and Q.z_dr_0p05_0p1 > 0.0393:
        z += -37.4 * (0.0573 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.0393)
    return max(0.0, z)


def neuron_2(Q):
    z = -2.28
    if Q.LHA >= 0.109:
        z += -9.2 * Q.LHA + 1.0028
    if Q.e2_sq < 0.000796:
        z += 1633.0 * Q.e2_sq - 5.13019
    if 0.000796 <= Q.e2_sq < 0.00785:
        z += 543.0 * Q.e2_sq - 4.26255
    if Q.girth < 0.00817:
        z += 269.0 * Q.girth - 2.19773
    if Q.log_sum_pt < 6.84:
        z += -7.53 * Q.log_sum_pt + 51.5052
    if Q.log_sum_pt >= 6.9:
        z += 39.5 * Q.log_sum_pt - 272.55
    z += 0.113 * Q.pt_7
    if 827.0 <= Q.sum_pt < 988.0:
        z += 0.00606 * Q.sum_pt - 5.01162
    if Q.sum_pt >= 988.0:
        z += -0.02304 * Q.sum_pt + 23.73918
    if Q.width < 0.00875:
        z += -575.0 * Q.width + 5.03125
    if Q.z_7 >= 0.0262:
        z += -51.5 * Q.z_7 + 1.3493
    if Q.lam1 < 0.00707 and Q.max_dr < 0.272:
        z += 1370.0 * (0.00707 - Q.lam1) * (0.272 - Q.max_dr)
    if Q.lam1 < 0.0084 and Q.pt_6 < 60.6:
        z += -2.67 * (0.0084 - Q.lam1) * (60.6 - Q.pt_6)
    if Q.log_sum_pt > 6.92 and Q.D2_b2 < 2.49:
        z += -7.25 * (Q.log_sum_pt - 6.92) * (2.49 - Q.D2_b2)
    if Q.log_sum_pt < 6.64 and Q.D2_b2 < 1.08:
        z += 9.42 * (6.64 - Q.log_sum_pt) * (1.08 - Q.D2_b2)
    if Q.log_sum_pt < 6.77 and Q.D2_b2 < 1.06:
        z += -7.72 * (6.77 - Q.log_sum_pt) * (1.06 - Q.D2_b2)
    if Q.sum_pt_top5 > 800.0 and Q.D2_b2 < 1.85:
        z += 0.00461 * (Q.sum_pt_top5 - 800.0) * (1.85 - Q.D2_b2)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.244
    if Q.LHA >= 0.316:
        z += 40.6 * Q.LHA - 12.8296
    if 0.00702 <= Q.e2_sq < 0.0111:
        z += 587.0 * Q.e2_sq - 4.12074
    if Q.e2_sq >= 0.0111:
        z += 83.0 * Q.e2_sq + 1.47366
    if Q.girth >= 0.0992:
        z += -56.5 * Q.girth + 5.6048
    if Q.girth2 < 0.00778:
        z += 1140.0 * Q.girth2 - 8.8692
    if Q.lam1 < 0.00733:
        z += -664.0 * Q.lam1 + 4.86712
    if 6.43 <= Q.log_sum_pt < 6.89:
        z += -2.04 * Q.log_sum_pt + 13.1172
    if Q.log_sum_pt >= 6.89:
        z += 3.69 * Q.log_sum_pt - 26.3625
    if Q.sj3_dr_max < 0.189:
        z += -6.4 * Q.sj3_dr_max - 0.4272
    if 0.189 <= Q.sj3_dr_max < 0.251:
        z += 26.4 * Q.sj3_dr_max - 6.6264
    if Q.sum_pt < 617.0:
        z += -0.00448 * Q.sum_pt + 2.76416
    if Q.tau1 < 0.112:
        z += -25.1 * Q.tau1 + 2.8112
    if Q.tau21_b2 < 0.319:
        z += -3.52 * Q.tau21_b2 + 1.12288
    if Q.sj2_dr > 0.0981 and Q.eccentricity > 0.968:
        z += 149.0 * (Q.sj2_dr - 0.0981) * (Q.eccentricity - 0.968)
    if Q.sj2_dr > 0.112 and Q.girth2_top2 < 0.00265:
        z += -2550.0 * (Q.sj2_dr - 0.112) * (0.00265 - Q.girth2_top2)
    if Q.width > 0.00442 and Q.pt_6 > 32.1:
        z += -3.33 * (Q.width - 0.00442) * (Q.pt_6 - 32.1)
    return max(0.0, z)


def neuron_4(Q):
    z = -5.87
    if Q.C2_b2 < 0.0247:
        z += -229.0 * Q.C2_b2 + 5.6563
    if Q.LHA < 0.416:
        z += -12.7 * Q.LHA + 5.2832
    if Q.N2 < 0.238:
        z += -8.89 * Q.N2 + 2.11582
    if Q.centroid_offset < 0.018:
        z += 9.16 * Q.centroid_offset - 0.16488
    if Q.e2 >= 0.0187:
        z += 151.0 * Q.e2 - 2.8237
    if Q.e2_sq < 9e-05:
        z += 3935.0 * Q.e2_sq + 7.6083
    if 9e-05 <= Q.e2_sq < 0.0159:
        z += -745.0 * Q.e2_sq + 8.0295
    if Q.e2_sq >= 0.0159:
        z += -240.0 * Q.e2_sq
    if Q.e3 < 0.000198:
        z += -32000.0 * Q.e3 + 6.4
    if 0.000198 <= Q.e3 < 0.0002:
        z += -35680.0 * Q.e3 + 7.12864
    if Q.e3 >= 0.0002:
        z += -3680.0 * Q.e3 + 0.72864
    if Q.girth < 0.0159:
        z += 40.0 * Q.girth - 0.636
    if Q.girth2 < 0.00202:
        z += 293.0 * Q.girth2 - 4.5109
    if 0.00202 <= Q.girth2 < 0.00864:
        z += 592.0 * Q.girth2 - 5.11488
    if Q.lam1 < 0.000149:
        z += 5040.0 * Q.lam1 - 0.75096
    if Q.lam2 < 0.000288:
        z += 4640.0 * Q.lam2 - 3.42386
    if 0.000288 <= Q.lam2 < 0.00125:
        z += 2170.0 * Q.lam2 - 2.7125
    if Q.log_sum_pt >= 6.76:
        z += 5.89 * Q.log_sum_pt - 39.8164
    if Q.max_dr < 0.0333:
        z += -13.1 * Q.max_dr + 0.43623
    if Q.max_dr >= 0.126:
        z += 19.5 * Q.max_dr - 2.457
    if Q.mean_eta < -0.0236:
        z += -49.2 * Q.mean_eta - 1.16112
    if Q.n_dr_0p1_0p2 >= 1.16:
        z += 0.166 * Q.n_dr_0p1_0p2 - 0.19256
    if Q.sj2_dr >= 0.205:
        z += -16.8 * Q.sj2_dr + 3.444
    if Q.sj3_dr_min >= 0.0771:
        z += -9.66 * Q.sj3_dr_min + 0.744786
    if Q.sum_pt_top3 < 580.0:
        z += -0.002 * Q.sum_pt_top3 + 1.16
    if Q.sum_pt_top3 >= 585.0:
        z += -0.00404 * Q.sum_pt_top3 + 2.3634
    if Q.width < 0.00791:
        z += 989.0 * Q.width - 11.07973
    if 0.00791 <= Q.width < 0.0179:
        z += 326.0 * Q.width - 5.8354
    if Q.N2 < 0.271 and Q.mean_phi < -0.0195:
        z += 312.0 * (0.271 - Q.N2) * (-0.0195 - Q.mean_phi)
    if Q.N2 < 0.239 and Q.pt_7 < 55.4:
        z += -0.323 * (0.239 - Q.N2) * (55.4 - Q.pt_7)
    if Q.e3 < 0.000183 and Q.sum_pt < 841.0:
        z += -51.8 * (0.000183 - Q.e3) * (841.0 - Q.sum_pt)
    if Q.girth2 < 0.00739 and Q.log_sum_pt < 6.7:
        z += 1770.0 * (0.00739 - Q.girth2) * (6.7 - Q.log_sum_pt)
    if Q.girth2_top2 < 0.00563 and Q.centroid_offset > 0.00501:
        z += 13700.0 * (0.00563 - Q.girth2_top2) * (Q.centroid_offset - 0.00501)
    if Q.sj2_dr > 0.236 and Q.D2_b2 < 1.61:
        z += -16.1 * (Q.sj2_dr - 0.236) * (1.61 - Q.D2_b2)
    if Q.sj3_dr_max > 0.38 and Q.D2_b2 < 1.69:
        z += 24.2 * (Q.sj3_dr_max - 0.38) * (1.69 - Q.D2_b2)
    if Q.sj3_dr_max < 0.35 and Q.log_sum_pt < 6.81:
        z += -33.0 * (0.35 - Q.sj3_dr_max) * (6.81 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_5(Q):
    z = -1.72
    if Q.centroid_offset < 0.0338:
        z += 35.0 * Q.centroid_offset - 1.183
    if Q.log_sum_pt >= 6.59:
        z += 4.29 * Q.log_sum_pt - 28.2711
    if Q.pt_6 < 28.3:
        z += -0.0858 * Q.pt_6 + 2.42814
    if Q.pt_7 < 45.0:
        z += -0.115 * Q.pt_7 + 5.428
    if 45.0 <= Q.pt_7 < 47.2:
        z += -0.1999 * Q.pt_7 + 9.2485
    if Q.pt_7 >= 47.2:
        z += -0.0849 * Q.pt_7 + 3.8205
    if Q.sum_pt < 730.0:
        z += 0.0109 * Q.sum_pt - 7.957
    if Q.z_7 >= 0.0273:
        z += 65.3 * Q.z_7 - 1.78269
    if Q.LHA < 0.173 and Q.log_sum_pt < 6.87:
        z += -63.4 * (0.173 - Q.LHA) * (6.87 - Q.log_sum_pt)
    if Q.e2 < 0.0516 and Q.log_sum_pt > 6.87:
        z += -252.0 * (0.0516 - Q.e2) * (Q.log_sum_pt - 6.87)
    if Q.e2 < 0.0356 and Q.z_7 < 0.0714:
        z += 1480.0 * (0.0356 - Q.e2) * (0.0714 - Q.z_7)
    if Q.girth2 < 0.00278 and Q.centroid_offset < 0.019:
        z += 80600.0 * (0.00278 - Q.girth2) * (0.019 - Q.centroid_offset)
    if Q.girth2 < 0.000994 and Q.n_dr_0p2_0p4 < 2.46:
        z += 347.0 * (0.000994 - Q.girth2) * (2.46 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.547
    if Q.C2_b2 < 0.00668:
        z += -213.0 * Q.C2_b2 + 1.42284
    if Q.C2_b2 >= 0.00696:
        z += -30.5 * Q.C2_b2 + 0.21228
    if Q.centroid_offset >= 0.0126:
        z += 42.3 * Q.centroid_offset - 0.53298
    if Q.e2_sq < 0.00558:
        z += -568.0 * Q.e2_sq + 3.16944
    if Q.girth2 < 0.00951:
        z += 365.0 * Q.girth2 - 3.47115
    if Q.log_sum_pt < 6.34:
        z += -12.1 * Q.log_sum_pt + 76.714
    if 6.35 <= Q.log_sum_pt < 6.67:
        z += -8.62 * Q.log_sum_pt + 54.737
    if Q.log_sum_pt >= 6.67:
        z += -0.63 * Q.log_sum_pt + 1.4437
    if Q.max_dr < 0.191:
        z += 13.8 * Q.max_dr - 2.6358
    if Q.pt_6 < 31.8:
        z += -0.0871 * Q.pt_6 + 2.76978
    z += 0.136 * Q.pt_7
    if Q.sj2_dr < 0.136:
        z += -9.26 * Q.sj2_dr + 1.25936
    if Q.sum_pt >= 967.0:
        z += 0.00605 * Q.sum_pt - 5.85035
    if Q.sum_pt_top5 >= 676.0:
        z += -0.00757 * Q.sum_pt_top5 + 5.11732
    if Q.tau1 < 0.0979:
        z += -30.4 * Q.tau1 + 2.97616
    if Q.width < 0.00597:
        z += 711.0 * Q.width - 5.57403
    if 0.00597 <= Q.width < 0.00613:
        z += 232.0 * Q.width - 2.7144
    if 0.00613 <= Q.width < 0.0117:
        z += 118.0 * Q.width - 2.01558
    if Q.width >= 0.0117:
        z += -114.0 * Q.width + 0.69882
    if Q.z_7 < 0.0397:
        z += -132.0 * Q.z_7 + 5.346
    if 0.0397 <= Q.z_7 < 0.0405:
        z += -240.0 * Q.z_7 + 9.6336
    if Q.z_7 >= 0.0405:
        z += -108.0 * Q.z_7 + 4.2876
    if Q.LHA > 0.234 and Q.eccentricity > 0.873:
        z += 140.0 * (Q.LHA - 0.234) * (Q.eccentricity - 0.873)
    if Q.centroid_offset > 0.00872 and Q.mean_phi2 < 0.0119:
        z += 1720.0 * (Q.centroid_offset - 0.00872) * (0.0119 - Q.mean_phi2)
    if Q.centroid_offset > 0.00913 and Q.sum_pt < 1050.0:
        z += -0.0613 * (Q.centroid_offset - 0.00913) * (1050.0 - Q.sum_pt)
    if Q.sj3_dr_max > 0.0711 and Q.eccentricity > 0.864:
        z += -192.0 * (Q.sj3_dr_max - 0.0711) * (Q.eccentricity - 0.864)
    if Q.sj3_dr_max > 0.167 and Q.eccentricity > 0.855:
        z += 229.0 * (Q.sj3_dr_max - 0.167) * (Q.eccentricity - 0.855)
    return max(0.0, z)


def neuron_7(Q):
    z = -1.35
    if Q.LHA >= 0.248:
        z += 19.5 * Q.LHA - 4.836
    if Q.e2 >= 0.0073:
        z += -65.4 * Q.e2 + 0.47742
    if Q.e2_sq < 0.0057:
        z += -14.0 * Q.e2_sq - 8.4894
    if 0.0057 <= Q.e2_sq < 0.0168:
        z += 772.0 * Q.e2_sq - 12.9696
    if Q.e3 < 8.88e-05:
        z += 12100.0 * Q.e3 - 2.2506
    if 8.88e-05 <= Q.e3 < 0.000186:
        z += 18960.0 * Q.e3 - 2.859768
    if Q.e3 >= 0.000186:
        z += 6860.0 * Q.e3 - 0.609168
    if Q.girth < 0.0171:
        z += 96.8 * Q.girth - 8.5184
    if 0.0171 <= Q.girth < 0.088:
        z += -20.2 * Q.girth - 6.5177
    if Q.girth >= 0.088:
        z += -117.0 * Q.girth + 2.0007
    if Q.girth2 < 0.0187:
        z += -1580.0 * Q.girth2 + 29.546
    if Q.lam2 >= 0.00064:
        z += -394.0 * Q.lam2 + 0.25216
    if Q.sj2_dr < 0.158:
        z += -7.1 * Q.sj2_dr + 0.1818
    if 0.158 <= Q.sj2_dr < 0.183:
        z += 37.6 * Q.sj2_dr - 6.8808
    if Q.sj3_dr_max >= 0.276:
        z += -8.75 * Q.sj3_dr_max + 2.415
    if Q.sum_pt < 735.0:
        z += -0.000291 * Q.sum_pt + 0.213885
    if Q.tau1 < 0.0139:
        z += -66.0 * Q.tau1 + 6.3294
    if 0.0139 <= Q.tau1 < 0.0959:
        z += 28.1 * Q.tau1 + 5.02141
    if Q.tau1 >= 0.0959:
        z += 94.1 * Q.tau1 - 1.30799
    if Q.width < 0.0042:
        z += 2290.0 * Q.width - 16.0413
    if 0.0042 <= Q.width < 0.00605:
        z += 1888.0 * Q.width - 14.3529
    if 0.00605 <= Q.width < 0.00759:
        z += 1018.0 * Q.width - 9.0894
    if Q.width >= 0.00759:
        z += -402.0 * Q.width + 1.6884
    if Q.e2_sq < 0.0136 and Q.pt_7 < 46.4:
        z += -1.92 * (0.0136 - Q.e2_sq) * (46.4 - Q.pt_7)
    if Q.girth2 > 0.0128 and Q.eccentricity > 0.952:
        z += 5540.0 * (Q.girth2 - 0.0128) * (Q.eccentricity - 0.952)
    if Q.girth2 > 0.00287 and Q.planar_flow < 0.225:
        z += 1010.0 * (Q.girth2 - 0.00287) * (0.225 - Q.planar_flow)
    if Q.planar_flow < 0.196 and Q.log_sum_pt > 6.48:
        z += 21.9 * (0.196 - Q.planar_flow) * (Q.log_sum_pt - 6.48)
    if Q.planar_flow < 0.225 and Q.z_7 < 0.0414:
        z += -214.0 * (0.225 - Q.planar_flow) * (0.0414 - Q.z_7)
    if Q.sj3_dr_max < 0.214 and Q.eccentricity > 0.944:
        z += -122.0 * (0.214 - Q.sj3_dr_max) * (Q.eccentricity - 0.944)
    if Q.tau1 < 0.0736 and Q.z_dr_0p05_0p1 > 0.489:
        z += -99.8 * (0.0736 - Q.tau1) * (Q.z_dr_0p05_0p1 - 0.489)
    if Q.width < 0.00635 and Q.mean_eta < -0.00704:
        z += 24100.0 * (0.00635 - Q.width) * (-0.00704 - Q.mean_eta)
    if Q.width < 0.00655 and Q.mean_eta > 0.00792:
        z += 21700.0 * (0.00655 - Q.width) * (Q.mean_eta - 0.00792)
    if Q.width < 0.0067 and Q.mean_phi > 0.00994:
        z += 22900.0 * (0.0067 - Q.width) * (Q.mean_phi - 0.00994)
    if Q.width < 0.00725 and Q.mean_phi < -0.00506:
        z += 17700.0 * (0.00725 - Q.width) * (-0.00506 - Q.mean_phi)
    return max(0.0, z)


def neuron_8(Q):
    z = -1.47
    if Q.C2 < 0.0202:
        z += 52.8 * Q.C2 - 1.43776
    if 0.0202 <= Q.C2 < 0.0362:
        z += 23.2 * Q.C2 - 0.83984
    if Q.N2 < 0.118:
        z += -7.68 * Q.N2 + 1.11264
    if 0.118 <= Q.N2 < 0.214:
        z += -2.15 * Q.N2 + 0.4601
    if Q.centroid_offset < 0.00814:
        z += -13.4 * Q.centroid_offset + 0.109076
    if Q.centroid_offset >= 0.0343:
        z += 17.6 * Q.centroid_offset - 0.60368
    if Q.e2_sq < 9.35e-05:
        z += 359.0 * Q.e2_sq - 2.56685
    if 9.35e-05 <= Q.e2_sq < 0.00715:
        z += 466.0 * Q.e2_sq - 2.576855
    if 0.00715 <= Q.e2_sq < 0.0164:
        z += 107.0 * Q.e2_sq - 0.0100045
    if Q.e2_sq >= 0.0164:
        z += 134.2 * Q.e2_sq - 0.4560845
    if Q.girth < 0.0219:
        z += 76.0 * Q.girth - 1.6644
    if 0.0618 <= Q.girth < 0.124:
        z += -17.9 * Q.girth + 1.10622
    if Q.girth >= 0.124:
        z += -0.1 * Q.girth - 1.10098
    if Q.girth2 < 0.000334:
        z += -2160.0 * Q.girth2 + 7.83744
    if 0.000334 <= Q.girth2 < 0.00745:
        z += -1000.0 * Q.girth2 + 7.45
    if Q.girth2_top2 >= 0.0146:
        z += -6.06 * Q.girth2_top2 + 0.088476
    if Q.girth2_top3 < 0.00732:
        z += 60.3 * Q.girth2_top3 - 0.441396
    if Q.lam1 < 0.000518:
        z += -633.0 * Q.lam1 - 4.72204
    if 0.000518 <= Q.lam1 < 0.00426:
        z += 1017.0 * Q.lam1 - 5.57674
    if 0.00426 <= Q.lam1 < 0.00734:
        z += 404.0 * Q.lam1 - 2.96536
    if Q.lam2 < 5.94e-05:
        z += -8205.0 * Q.lam2 + 0.454424
    if 5.94e-05 <= Q.lam2 < 0.000188:
        z += -2545.0 * Q.lam2 + 0.11822
    if 0.000188 <= Q.lam2 < 0.0011:
        z += 395.0 * Q.lam2 - 0.4345
    if Q.log_sum_pt >= 6.7:
        z += -13.4 * Q.log_sum_pt + 89.78
    if Q.max_dr < 0.178:
        z += -4.85 * Q.max_dr + 0.8633
    if Q.mean_eta >= 0.00151:
        z += 3.91 * Q.mean_eta - 0.0059041
    if Q.mean_phi < -0.011:
        z += -4.7 * Q.mean_phi - 0.266512
    if -0.011 <= Q.mean_phi < -0.000447:
        z += 15.6 * Q.mean_phi - 0.043212
    if -0.000447 <= Q.mean_phi < 0.00277:
        z += 20.37 * Q.mean_phi - 0.04107981
    if Q.mean_phi >= 0.00277:
        z += 4.77 * Q.mean_phi + 0.00213219
    if Q.planar_flow < 0.038:
        z += 10.2 * Q.planar_flow - 0.3876
    if Q.pt_6 < 29.9:
        z += 0.0194 * Q.pt_6 - 0.58006
    if Q.pt_7 < 47.4:
        z += -0.00707 * Q.pt_7 + 0.335118
    if Q.sj2_dr < 0.161:
        z += 2.57 * Q.sj2_dr - 0.41377
    if Q.sj3_dr_max < 0.0993:
        z += -0.064 * Q.sj3_dr_max + 0.511791
    if 0.0993 <= Q.sj3_dr_max < 0.138:
        z += 7.966 * Q.sj3_dr_max - 0.285588
    if 0.138 <= Q.sj3_dr_max < 0.198:
        z += -4.534 * Q.sj3_dr_max + 1.439412
    if 0.198 <= Q.sj3_dr_max < 0.346:
        z += -3.66 * Q.sj3_dr_max + 1.26636
    if Q.sj3_dr_min < 0.0486:
        z += 3.57 * Q.sj3_dr_min - 0.173502
    if Q.sum_pt_top5 >= 690.0:
        z += 0.0134 * Q.sum_pt_top5 - 9.246
    if Q.width < 0.00435:
        z += -1650.0 * Q.width + 7.1775
    if Q.width >= 0.0184:
        z += -107.0 * Q.width + 1.9688
    if Q.girth < 0.0553 and Q.width < 0.00561:
        z += -17000.0 * (0.0553 - Q.girth) * (0.00561 - Q.width)
    if Q.girth2 < 0.00507 and Q.centroid_offset > 0.00798:
        z += -21400.0 * (0.00507 - Q.girth2) * (Q.centroid_offset - 0.00798)
    if Q.girth2 < 0.00652 and Q.centroid_offset < 0.0245:
        z += 7120.0 * (0.00652 - Q.girth2) * (0.0245 - Q.centroid_offset)
    if Q.girth2 < 0.00491 and Q.log_sum_pt < 6.55:
        z += -399.0 * (0.00491 - Q.girth2) * (6.55 - Q.log_sum_pt)
    if Q.girth2 < 0.00544 and Q.pt_7 < 44.2:
        z += -6.23 * (0.00544 - Q.girth2) * (44.2 - Q.pt_7)
    if Q.log_sum_pt > 6.7 and Q.width < 0.00632:
        z += 1520.0 * (Q.log_sum_pt - 6.7) * (0.00632 - Q.width)
    if Q.sj3_dr_max < 0.166 and Q.width > 0.00264:
        z += 2270.0 * (0.166 - Q.sj3_dr_max) * (Q.width - 0.00264)
    if Q.sum_pt_top5 > 691.0 and Q.girth2_top3 < 0.00966:
        z += -1.15 * (Q.sum_pt_top5 - 691.0) * (0.00966 - Q.girth2_top3)
    return max(0.0, z)


def neuron_9(Q):
    z = -3.39
    if Q.girth2 < 0.00712:
        z += -1910.0 * Q.girth2 + 13.5992
    if Q.lam2 >= 0.000442:
        z += 1280.0 * Q.lam2 - 0.56576
    if Q.log_sum_pt < 6.86:
        z += -4.34 * Q.log_sum_pt + 29.7724
    if Q.sj3_dr_max < 0.122:
        z += 19.5 * Q.sj3_dr_max - 2.379
    if Q.sj3_dr_max >= 0.309:
        z += 5.77 * Q.sj3_dr_max - 1.78293
    if Q.tau1 < 0.0903:
        z += 37.9 * Q.tau1 - 3.42237
    if Q.centroid_offset < 0.0197 and Q.C2 < 0.0643:
        z += 3480.0 * (0.0197 - Q.centroid_offset) * (0.0643 - Q.C2)
    if Q.centroid_offset < 0.0181 and Q.D2_b2 < 0.336:
        z += -395.0 * (0.0181 - Q.centroid_offset) * (0.336 - Q.D2_b2)
    if Q.girth < 0.0677 and Q.mean_phi < -0.00128:
        z += -3170.0 * (0.0677 - Q.girth) * (-0.00128 - Q.mean_phi)
    if Q.girth < 0.07 and Q.mean_phi > 0.00476:
        z += -2980.0 * (0.07 - Q.girth) * (Q.mean_phi - 0.00476)
    if Q.lam2 > 0.000494 and Q.planar_flow > 0.122:
        z += -879.0 * (Q.lam2 - 0.000494) * (Q.planar_flow - 0.122)
    if Q.log_sum_pt < 6.83 and Q.planar_flow > 0.0176:
        z += -3.55 * (6.83 - Q.log_sum_pt) * (Q.planar_flow - 0.0176)
    if Q.width < 0.00608 and Q.mean_eta > 0.00407:
        z += -26500.0 * (0.00608 - Q.width) * (Q.mean_eta - 0.00407)
    if Q.width < 0.0062 and Q.mean_eta < -0.00284:
        z += -24000.0 * (0.0062 - Q.width) * (-0.00284 - Q.mean_eta)
    return max(0.0, z)


def neuron_10(Q):
    z = 11.8
    if Q.C2 >= 0.0499:
        z += 28.5 * Q.C2 - 1.42215
    if Q.LHA >= 0.209:
        z += -7.86 * Q.LHA + 1.64274
    if Q.lam1 < 0.00417:
        z += 726.0 * Q.lam1 - 11.6886
    if 0.00417 <= Q.lam1 < 0.0159:
        z += 294.0 * Q.lam1 - 9.88716
    if 0.0159 <= Q.lam1 < 0.0161:
        z += 993.0 * Q.lam1 - 21.00126
    if Q.lam1 >= 0.0161:
        z += 267.0 * Q.lam1 - 9.31266
    if Q.lam2 >= 0.000281:
        z += 636.0 * Q.lam2 - 0.178716
    if Q.z_6 < 0.0645:
        z += 8.36 * Q.z_6 - 0.53922
    if Q.LHA > 0.312 and Q.planar_flow < 0.503:
        z += -53.3 * (Q.LHA - 0.312) * (0.503 - Q.planar_flow)
    if Q.e3 > -7.29e-05 and Q.pt_7 < 32.6:
        z += -201.0 * (Q.e3 - -7.29e-05) * (32.6 - Q.pt_7)
    if Q.lam2 > 4.09e-05 and Q.log_sum_pt < 6.69:
        z += -587.0 * (Q.lam2 - 4.09e-05) * (6.69 - Q.log_sum_pt)
    if Q.pt_6 < 46.8 and Q.log_sum_pt < 6.55:
        z += -0.108 * (46.8 - Q.pt_6) * (6.55 - Q.log_sum_pt)
    if Q.sj2_dr > 0.166 and Q.planar_flow < 0.71:
        z += -18.2 * (Q.sj2_dr - 0.166) * (0.71 - Q.planar_flow)
    return max(0.0, z)


def neuron_11(Q):
    z = 0.347
    if Q.C2_b2 < 0.00944:
        z += 177.0 * Q.C2_b2 - 1.67088
    if Q.e2_sq < 0.00841:
        z += 1820.0 * Q.e2_sq - 15.3062
    if Q.girth < 0.077:
        z += 84.1 * Q.girth - 6.4757
    if Q.lam2 < 0.00126:
        z += -1660.0 * Q.lam2 + 2.0916
    if Q.planar_flow < 0.274:
        z += -6.61 * Q.planar_flow + 1.81114
    if Q.pt_7 < 40.2:
        z += 0.0586 * Q.pt_7 - 2.35572
    if Q.sj3_dr_max >= 0.174:
        z += -14.0 * Q.sj3_dr_max + 2.436
    if Q.tau1 < 0.0546:
        z += -37.5 * Q.tau1 + 2.0475
    if Q.width < 0.00457:
        z += -1695.0 * Q.width + 20.29298
    if 0.00457 <= Q.width < 0.00897:
        z += -2411.0 * Q.width + 23.5651
    if 0.00897 <= Q.width < 0.013:
        z += -481.0 * Q.width + 6.253
    if Q.centroid_offset < 0.0165 and Q.sj3_dr_min < 0.0801:
        z += -1470.0 * (0.0165 - Q.centroid_offset) * (0.0801 - Q.sj3_dr_min)
    if Q.centroid_offset < 0.0166 and Q.tau21_b2 < 0.0549:
        z += 2560.0 * (0.0166 - Q.centroid_offset) * (0.0549 - Q.tau21_b2)
    if Q.log_sum_pt > 6.46 and Q.z_7 < 0.0543:
        z += 42.7 * (Q.log_sum_pt - 6.46) * (0.0543 - Q.z_7)
    if Q.planar_flow < 0.265 and Q.e3 < 6.99e-06:
        z += -774000.0 * (0.265 - Q.planar_flow) * (6.99e-06 - Q.e3)
    if Q.planar_flow < 0.286 and Q.sum_pt < 861.0:
        z += -0.0178 * (0.286 - Q.planar_flow) * (861.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_12(Q):
    z = -2.9
    if 0.00287 <= Q.centroid_offset < 0.0231:
        z += 23.2 * Q.centroid_offset - 0.066584
    if 0.0231 <= Q.centroid_offset < 0.0554:
        z += 39.2 * Q.centroid_offset - 0.436184
    if Q.centroid_offset >= 0.0554:
        z += 50.3 * Q.centroid_offset - 1.051124
    if Q.e2 < 0.0573:
        z += 15.2 * Q.e2 - 0.87096
    if 0.00587 <= Q.e2_sq < 0.0194:
        z += 171.0 * Q.e2_sq - 1.00377
    if Q.e2_sq >= 0.0194:
        z += 257.4 * Q.e2_sq - 2.67993
    if Q.e3 < 0.000272:
        z += -3370.0 * Q.e3 + 0.91664
    if Q.girth >= 0.146:
        z += 21.2 * Q.girth - 3.0952
    if Q.girth2 < 0.0132:
        z += 83.5 * Q.girth2 - 1.1022
    if Q.lam1 >= 0.00624:
        z += -48.4 * Q.lam1 + 0.302016
    if Q.log_sum_pt >= 6.7:
        z += 7.31 * Q.log_sum_pt - 48.977
    if Q.mean_phi < -0.0164:
        z += 11.3 * Q.mean_phi + 0.18532
    if Q.n_dr_0p1_0p2 < 3.64:
        z += 0.02 * Q.n_dr_0p1_0p2 - 0.0728
    if Q.n_dr_0p2_0p4 >= 0.699:
        z += 0.518 * Q.n_dr_0p2_0p4 - 0.362082
    if Q.pt_7 >= 17.5:
        z += 0.0346 * Q.pt_7 - 0.6055
    if Q.ptdr0_4 >= 17.1:
        z += 0.112 * Q.ptdr0_4 - 1.9152
    if 0.144 <= Q.sj2_dr < 0.266:
        z += -1.6 * Q.sj2_dr + 0.2304
    if Q.sj2_dr >= 0.266:
        z += 0.67 * Q.sj2_dr - 0.37342
    if Q.sj3_dr_max >= 0.291:
        z += 2.08 * Q.sj3_dr_max - 0.60528
    if Q.sum_pt < 545.0:
        z += 0.0029 * Q.sum_pt - 1.5805
    if Q.tau1 >= 0.0357:
        z += -14.1 * Q.tau1 + 0.50337
    if Q.width < 0.00406:
        z += 1010.0 * Q.width - 4.1006
    if Q.width >= 0.0179:
        z += 185.0 * Q.width - 3.3115
    if Q.z_dr_0p2_0p4 >= 0.13:
        z += -3.47 * Q.z_dr_0p2_0p4 + 0.4511
    if Q.e2 > 0.0722 and Q.sum_pt > 606.0:
        z += -0.251 * (Q.e2 - 0.0722) * (Q.sum_pt - 606.0)
    if Q.girth2 > 0.0162 and Q.planar_flow < 0.684:
        z += -90.9 * (Q.girth2 - 0.0162) * (0.684 - Q.planar_flow)
    if Q.girth2 > 0.0178 and Q.pt_7 < 52.1:
        z += -2.36 * (Q.girth2 - 0.0178) * (52.1 - Q.pt_7)
    if Q.girth2 > 0.0207 and Q.sum_pt < 816.0:
        z += -0.355 * (Q.girth2 - 0.0207) * (816.0 - Q.sum_pt)
    if Q.girth2_top2 > 0.0289 and Q.sum_pt > 573.0:
        z += -0.32 * (Q.girth2_top2 - 0.0289) * (Q.sum_pt - 573.0)
    if Q.ptdr0_4 > 14.6 and Q.sum_pt < 820.0:
        z += 0.000719 * (Q.ptdr0_4 - 14.6) * (820.0 - Q.sum_pt)
    if Q.ptdr0_4 > 14.9 and Q.sum_pt < 967.0:
        z += -0.000659 * (Q.ptdr0_4 - 14.9) * (967.0 - Q.sum_pt)
    if Q.width > 0.0103 and Q.log_sum_pt > 6.45:
        z += 491.0 * (Q.width - 0.0103) * (Q.log_sum_pt - 6.45)
    return max(0.0, z)


def neuron_13(Q):
    z = 5.88
    if Q.LHA >= 0.18:
        z += -22.8 * Q.LHA + 4.104
    if Q.girth < 0.0254:
        z += -63.4 * Q.girth + 1.61036
    if Q.lam1 < 0.0156:
        z += -393.0 * Q.lam1 + 6.1308
    if Q.lam1 >= 0.017:
        z += -90.8 * Q.lam1 + 1.5436
    if Q.lam2 < 0.000269:
        z += 1770.0 * Q.lam2 - 0.47613
    if Q.log_sum_pt >= 6.56:
        z += 4.08 * Q.log_sum_pt - 26.7648
    if Q.pt_6 >= 23.4:
        z += -0.0979 * Q.pt_6 + 2.29086
    if Q.sum_pt < 707.0:
        z += 0.00744 * Q.sum_pt - 5.26008
    if Q.sum_pt >= 1010.0:
        z += -0.00713 * Q.sum_pt + 7.2013
    if Q.tau1 < 0.111:
        z += 25.6 * Q.tau1 - 2.8416
    if Q.z_6 < 0.0438:
        z += 120.0 * Q.z_6 - 5.352
    if 0.0438 <= Q.z_6 < 0.0446:
        z += 194.3 * Q.z_6 - 8.60634
    if Q.z_6 >= 0.0446:
        z += 74.3 * Q.z_6 - 3.25434
    if Q.e3 < 7.24e-05 and Q.centroid_offset < 0.0374:
        z += -685000.0 * (7.24e-05 - Q.e3) * (0.0374 - Q.centroid_offset)
    if Q.girth < 0.134 and Q.log_sum_pt < 6.85:
        z += -69.3 * (0.134 - Q.girth) * (6.85 - Q.log_sum_pt)
    if Q.girth < 0.146 and Q.pt_7 < 42.9:
        z += -0.754 * (0.146 - Q.girth) * (42.9 - Q.pt_7)
    if Q.lam1 < 0.0208 and Q.pt_7 < 26.6:
        z += -9.17 * (0.0208 - Q.lam1) * (26.6 - Q.pt_7)
    if Q.sum_pt > 1020.0 and Q.z_6 > 0.0217:
        z += 0.264 * (Q.sum_pt - 1020.0) * (Q.z_6 - 0.0217)
    if Q.sum_pt_top5 > 599.0 and Q.pt_7 < 39.6:
        z += 0.000488 * (Q.sum_pt_top5 - 599.0) * (39.6 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = 4.23
    if Q.D2_b2 >= 1.21:
        z += 0.0228 * Q.D2_b2 - 0.027588
    if Q.LHA >= 0.313:
        z += 62.2 * Q.LHA - 19.4686
    if Q.centroid_offset >= 0.0282:
        z += -51.0 * Q.centroid_offset + 1.4382
    if Q.e2 < 0.0254:
        z += -80.2 * Q.e2 + 2.03708
    if Q.e2 >= 0.0417:
        z += 102.0 * Q.e2 - 4.2534
    if Q.e2_sq < 0.00209:
        z += 808.0 * Q.e2_sq - 13.6552
    if 0.00209 <= Q.e2_sq < 0.00612:
        z += -142.0 * Q.e2_sq - 11.6697
    if 0.00612 <= Q.e2_sq < 0.0169:
        z += 401.0 * Q.e2_sq - 14.99286
    if Q.e2_sq >= 0.0169:
        z += -407.0 * Q.e2_sq - 1.33766
    if Q.eccentricity >= 0.96:
        z += -403.0 * Q.eccentricity + 386.88
    if Q.girth < 0.0818:
        z += 63.6 * Q.girth - 5.20248
    if Q.girth >= 0.0891:
        z += -158.0 * Q.girth + 14.0778
    if Q.girth2 < 0.00731:
        z += 11.6445
    if 0.00731 <= Q.girth2 < 0.0184:
        z += -1050.0 * Q.girth2 + 19.32
    if Q.lam1 < 0.000414:
        z += 2570.0 * Q.lam1 - 1.06398
    if Q.log_sum_pt >= 6.8:
        z += -6.46 * Q.log_sum_pt + 43.928
    if Q.mean_phi < 0.00645:
        z += -7.84 * Q.mean_phi + 0.050568
    if Q.planar_flow < 0.148:
        z += -122.0 * Q.planar_flow + 18.056
    if Q.pt_7 >= 26.8:
        z += -0.0534 * Q.pt_7 + 1.43112
    if Q.sj2_dr < 0.152:
        z += -22.6 * Q.sj2_dr + 2.2553
    if 0.152 <= Q.sj2_dr < 0.221:
        z += 17.1 * Q.sj2_dr - 3.7791
    if Q.sj3_dr_max < 0.211:
        z += 18.3 * Q.sj3_dr_max - 3.8613
    if Q.sum_pt < 513.0:
        z += 0.00655 * Q.sum_pt - 3.36015
    if 0.0134 <= Q.tau1 < 0.0352:
        z += -76.8 * Q.tau1 + 1.02912
    if Q.tau1 >= 0.0352:
        z += -10.7 * Q.tau1 - 1.2976
    if Q.width < 0.0043:
        z += 464.0 * Q.width - 1.9952
    if Q.width >= 0.0129:
        z += 523.0 * Q.width - 6.7467
    if Q.z_7 >= 0.0245:
        z += 33.6 * Q.z_7 - 0.8232
    if Q.z_dr_0p05_0p1 >= 0.746:
        z += -8.93 * Q.z_dr_0p05_0p1 + 6.66178
    if Q.e2_sq > 0.002 and Q.log_sum_pt > 6.24:
        z += 936.0 * (Q.e2_sq - 0.002) * (Q.log_sum_pt - 6.24)
    if Q.girth2 > 0.00797 and Q.log_sum_pt > 6.25:
        z += -1490.0 * (Q.girth2 - 0.00797) * (Q.log_sum_pt - 6.25)
    if Q.planar_flow < 0.115 and Q.centroid_offset < 0.0182:
        z += -377.0 * (0.115 - Q.planar_flow) * (0.0182 - Q.centroid_offset)
    if Q.planar_flow < 0.113 and Q.lam1 < 0.0163:
        z += -1190.0 * (0.113 - Q.planar_flow) * (0.0163 - Q.lam1)
    if Q.z_dr_0p05_0p1 > 0.736 and Q.n_dr_0p2_0p4 < 0.973:
        z += 7.93 * (Q.z_dr_0p05_0p1 - 0.736) * (0.973 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_15(Q):
    z = -5.28
    if 0.243 <= Q.LHA < 0.318:
        z += 10.2 * Q.LHA - 2.4786
    if Q.LHA >= 0.318:
        z += 66.5 * Q.LHA - 20.382
    if Q.centroid_offset < 0.0213:
        z += 22.2 * Q.centroid_offset - 0.47286
    if Q.e2 < 0.0498:
        z += -75.6 * Q.e2 + 3.76488
    if Q.e2_sq < 0.00448:
        z += -370.0 * Q.e2_sq + 2.3828
    if 0.00448 <= Q.e2_sq < 0.00644:
        z += -74.0 * Q.e2_sq + 1.05672
    if 0.00644 <= Q.e2_sq < 0.0195:
        z += 296.0 * Q.e2_sq - 1.32608
    if Q.e2_sq >= 0.0195:
        z += 151.0 * Q.e2_sq + 1.50142
    if Q.e3 < 0.000163:
        z += 3380.0 * Q.e3 - 0.55094
    if Q.girth < 0.0244:
        z += -0.103 * Q.girth + 0.0025132
    if Q.girth >= 0.0839:
        z += -161.0 * Q.girth + 13.5079
    if Q.girth2 < 0.0076:
        z += 817.0 * Q.girth2 - 2.3977
    if 0.0076 <= Q.girth2 < 0.0181:
        z += -363.0 * Q.girth2 + 6.5703
    if Q.lam1 < 0.00735:
        z += -428.0 * Q.lam1 + 3.1458
    if Q.lam2 < 5.52e-05:
        z += 10575.0 * Q.lam2 + 0.85055
    if 5.52e-05 <= Q.lam2 < 0.000251:
        z += -425.0 * Q.lam2 + 1.45775
    if 0.000251 <= Q.lam2 < 0.00343:
        z += -450.6 * Q.lam2 + 1.464176
    if Q.lam2 >= 0.00343:
        z += -25.6 * Q.lam2 + 0.0064256
    if Q.log_sum_pt >= 6.82:
        z += 4.87 * Q.log_sum_pt - 33.2134
    if Q.mean_phi < 0.00889:
        z += -4.06 * Q.mean_phi + 0.0360934
    if Q.planar_flow < 0.0314:
        z += -18.1 * Q.planar_flow + 0.56834
    if Q.sj3_dr_max < 0.178:
        z += -1.9 * Q.sj3_dr_max - 0.3288
    if 0.178 <= Q.sj3_dr_max < 0.236:
        z += 11.5 * Q.sj3_dr_max - 2.714
    if Q.sj3_dr_max >= 0.326:
        z += -4.62 * Q.sj3_dr_max + 1.50612
    if Q.sum_pt < 757.0:
        z += 0.000765 * Q.sum_pt - 0.579105
    if Q.sum_pt_top3 >= 676.0:
        z += -0.00251 * Q.sum_pt_top3 + 1.69676
    if Q.tau1 < 0.0509:
        z += -2.5 * Q.tau1 - 0.5095
    if 0.0509 <= Q.tau1 < 0.104:
        z += 19.5 * Q.tau1 - 1.6293
    if 0.104 <= Q.tau1 < 0.113:
        z += 107.5 * Q.tau1 - 10.7813
    if Q.tau1 >= 0.113:
        z += 22.0 * Q.tau1 - 1.1198
    if Q.width < 0.00494:
        z += 433.0 * Q.width - 2.13902
    if Q.z_dr_0p05_0p1 >= 0.719:
        z += -5.42 * Q.z_dr_0p05_0p1 + 3.89698
    if Q.N2 < 0.186 and Q.LHA > 0.331:
        z += -230.0 * (0.186 - Q.N2) * (Q.LHA - 0.331)
    if Q.N2 < 0.206 and Q.LHA > 0.275:
        z += 155.0 * (0.206 - Q.N2) * (Q.LHA - 0.275)
    if Q.N2 < 0.252 and Q.girth2 > 0.00799:
        z += -1090.0 * (0.252 - Q.N2) * (Q.girth2 - 0.00799)
    if Q.N2 < 0.227 and Q.pt_7 < 24.2:
        z += -0.388 * (0.227 - Q.N2) * (24.2 - Q.pt_7)
    if Q.N2 < 0.228 and Q.sj2_dr < 0.189:
        z += -288.0 * (0.228 - Q.N2) * (0.189 - Q.sj2_dr)
    if Q.N2 < 0.23 and Q.sj2_dr < 0.161:
        z += 398.0 * (0.23 - Q.N2) * (0.161 - Q.sj2_dr)
    if Q.girth2 < 0.0133 and Q.centroid_offset > 0.0491:
        z += -19200.0 * (0.0133 - Q.girth2) * (Q.centroid_offset - 0.0491)
    if Q.girth2_top3 < 0.00164 and Q.eccentricity > 0.954:
        z += -19900.0 * (0.00164 - Q.girth2_top3) * (Q.eccentricity - 0.954)
    if Q.planar_flow < 0.215 and Q.sum_pt_top5 > 434.0:
        z += 0.015 * (0.215 - Q.planar_flow) * (Q.sum_pt_top5 - 434.0)
    if Q.z_dr_0p05_0p1 > 0.702 and Q.z_dr_0p2_0p4 < 0.0509:
        z += 92.8 * (Q.z_dr_0p05_0p1 - 0.702) * (0.0509 - Q.z_dr_0p2_0p4)
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
