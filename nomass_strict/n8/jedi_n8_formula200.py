"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; no mass observables or exact equivalents), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.3% (the network: 65.8%); same class as the network for 86.5% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
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
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
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
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        max_dr=max(dr[i] for i in real),
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
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = 3.02
    if Q.LHA >= 0.31:
        z += -75.6 * Q.LHA + 23.436
    if Q.centroid_offset >= 0.0061:
        z += -77.1 * Q.centroid_offset + 0.47031
    if Q.lam1 < 0.0041:
        z += 734.0 * Q.lam1 - 3.0094
    if Q.log_sum_pt >= 6.8:
        z += -25.9 * Q.log_sum_pt + 176.12
    if Q.planar_flow < 0.14:
        z += -12.2 * Q.planar_flow + 1.708
    if 0.13 <= Q.sj3_dr_max < 0.19:
        z += 19.8 * Q.sj3_dr_max - 2.574
    if Q.sj3_dr_max >= 0.19:
        z += -16.1 * Q.sj3_dr_max + 4.247
    if Q.tau2 < 0.017:
        z += 198.0 * Q.tau2 - 3.366
    if Q.e3 < 6.6e-05 and Q.sj3_z3 < 0.087:
        z += 416000.0 * (6.6e-05 - Q.e3) * (0.087 - Q.sj3_z3)
    if Q.lam1 < 0.0042 and Q.D2 < 0.78:
        z += -7060.0 * (0.0042 - Q.lam1) * (0.78 - Q.D2)
    if Q.lam1 < 0.0042 and Q.sum_pt < 730.0:
        z += 3.45 * (0.0042 - Q.lam1) * (730.0 - Q.sum_pt)
    if Q.log_sum_pt > 6.8 and Q.tau21_b2 < 0.048:
        z += -475.0 * (Q.log_sum_pt - 6.8) * (0.048 - Q.tau21_b2)
    if Q.planar_flow < 0.045 and Q.tau21_b2 < 0.061:
        z += 1100.0 * (0.045 - Q.planar_flow) * (0.061 - Q.tau21_b2)
    if Q.sum_pt < 700.0 and Q.centroid_offset > 0.012:
        z += 0.337 * (700.0 - Q.sum_pt) * (Q.centroid_offset - 0.012)
    if Q.sum_pt < 760.0 and Q.e3 < 0.00052:
        z += -16.6 * (760.0 - Q.sum_pt) * (0.00052 - Q.e3)
    if Q.sum_pt_top5 > 800.0 and Q.tau21_b2 < 0.14:
        z += 0.253 * (Q.sum_pt_top5 - 800.0) * (0.14 - Q.tau21_b2)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.39
    if 6.4 <= Q.log_sum_pt < 6.6:
        z += 6.84 * Q.log_sum_pt - 43.776
    if Q.log_sum_pt >= 6.6:
        z += 17.24 * Q.log_sum_pt - 112.416
    if Q.pt_7 >= 34.0:
        z += 0.122 * Q.pt_7 - 4.148
    if Q.sj3_dr_min >= 0.028:
        z += -16.6 * Q.sj3_dr_min + 0.4648
    if Q.tau1 < 0.1:
        z += -18.7 * Q.tau1 + 1.87
    if Q.z_7 < 0.065:
        z += 63.2 * Q.z_7 - 4.108
    if Q.lam1 < 0.0079 and Q.lam2 < 0.0015:
        z += -426000.0 * (0.0079 - Q.lam1) * (0.0015 - Q.lam2)
    if Q.lam1 < 0.0088 and Q.n_pt_above_50 > 6.1:
        z += -70.1 * (0.0088 - Q.lam1) * (Q.n_pt_above_50 - 6.1)
    if Q.log_sum_pt > 6.3 and Q.e3 < 1.3e-05:
        z += 198000.0 * (Q.log_sum_pt - 6.3) * (1.3e-05 - Q.e3)
    if Q.log_sum_pt > 6.6 and Q.girth2_top3 < 0.0075:
        z += -1390.0 * (Q.log_sum_pt - 6.6) * (0.0075 - Q.girth2_top3)
    if Q.pt_7 > 34.0 and Q.tau2 < 0.015:
        z += -7.88 * (Q.pt_7 - 34.0) * (0.015 - Q.tau2)
    if Q.sum_pt > 990.0 and Q.girth2_top2 < 0.0074:
        z += -2.59 * (Q.sum_pt - 990.0) * (0.0074 - Q.girth2_top2)
    return max(0.0, z)


def neuron_2(Q):
    z = 1.44
    if Q.M2 < 0.12:
        z += 16.9 * Q.M2 - 2.028
    if Q.girth < 0.0078:
        z += 550.0 * Q.girth - 0.3254
    if 0.0078 <= Q.girth < 0.1:
        z += -43.0 * Q.girth + 4.3
    if Q.log_sum_pt < 6.6:
        z += -9.6 * Q.log_sum_pt + 63.36
    if Q.mean_phi2 < 8.7e-05:
        z += -21300.0 * Q.mean_phi2 + 1.8531
    if Q.pt_7 < 55.0:
        z += 0.134 * Q.pt_7 - 7.37
    if Q.z_7 < 0.079:
        z += -58.9 * Q.z_7 + 4.7709
    if 0.079 <= Q.z_7 < 0.081:
        z += -121.8 * Q.z_7 + 9.74
    if Q.z_7 >= 0.081:
        z += -62.9 * Q.z_7 + 4.9691
    if Q.log_sum_pt < 6.6 and Q.D2_b2 > -0.04:
        z += -1.95 * (6.6 - Q.log_sum_pt) * (Q.D2_b2 - -0.04)
    if Q.log_sum_pt > 6.9 and Q.absphi_0 < 0.1:
        z += 59.3 * (Q.log_sum_pt - 6.9) * (0.1 - Q.absphi_0)
    if Q.log_sum_pt > 6.1 and Q.mean_phi2 < 8.9e-05:
        z += -35300.0 * (Q.log_sum_pt - 6.1) * (8.9e-05 - Q.mean_phi2)
    if Q.sum_pt_top5 < 750.0 and Q.D2_b2 > 0.18:
        z += 0.00265 * (750.0 - Q.sum_pt_top5) * (Q.D2_b2 - 0.18)
    if Q.tau1 < 0.043 and Q.planar_flow < 0.89:
        z += -44.7 * (0.043 - Q.tau1) * (0.89 - Q.planar_flow)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.418
    if 0.07 <= Q.girth < 0.099:
        z += 90.6 * Q.girth - 6.342
    if Q.girth >= 0.099:
        z += -29.4 * Q.girth + 5.538
    if Q.lam2 < 0.0032:
        z += 713.0 * Q.lam2 - 2.2816
    if Q.sd_rg >= 0.27:
        z += -38.2 * Q.sd_rg + 10.314
    if Q.sj2_dr >= 0.16:
        z += 28.6 * Q.sj2_dr - 4.576
    if Q.sj3_dr_max >= 0.22:
        z += 21.0 * Q.sj3_dr_max - 4.62
    if Q.max_dr > 0.25 and Q.z_dr_0_0p05 < 0.66:
        z += -83.3 * (Q.max_dr - 0.25) * (0.66 - Q.z_dr_0_0p05)
    if Q.sj2_dr > 0.16 and Q.C2 < 0.096:
        z += 468.0 * (Q.sj2_dr - 0.16) * (0.096 - Q.C2)
    if Q.sj2_dr > 0.17 and Q.log_sum_pt > 6.1:
        z += -37.7 * (Q.sj2_dr - 0.17) * (Q.log_sum_pt - 6.1)
    if Q.sj2_dr > 0.16 and Q.n_dr_0_0p05 > 1.6:
        z += -4.84 * (Q.sj2_dr - 0.16) * (Q.n_dr_0_0p05 - 1.6)
    return max(0.0, z)


def neuron_4(Q):
    z = -0.949
    if Q.e3 < 0.0002:
        z += -27500.0 * Q.e3 + 5.5
    if Q.lam1 < 0.0087:
        z += 867.0 * Q.lam1 - 7.5429
    if Q.lam2 < 0.00074:
        z += 5340.0 * Q.lam2 - 3.9516
    if Q.sj3_dr_min < 0.21:
        z += -66.1 * Q.sj3_dr_min + 13.881
    if Q.sum_pt < 840.0:
        z += 0.0074 * Q.sum_pt - 6.216
    if Q.dr_0 < 0.075 and Q.centroid_offset > 0.0011:
        z += 1750.0 * (0.075 - Q.dr_0) * (Q.centroid_offset - 0.0011)
    if Q.sj2_dr > 0.2 and Q.lam2 > 0.0012:
        z += -13900.0 * (Q.sj2_dr - 0.2) * (Q.lam2 - 0.0012)
    if Q.sj3_dr_max > 0.23 and Q.log_sum_pt > 6.3:
        z += -64.4 * (Q.sj3_dr_max - 0.23) * (Q.log_sum_pt - 6.3)
    if Q.sj3_dr_min < 0.2 and Q.lam2 < 0.0034:
        z += -12700.0 * (0.2 - Q.sj3_dr_min) * (0.0034 - Q.lam2)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.685
    if Q.centroid_offset < 0.029:
        z += 67.6 * Q.centroid_offset - 1.9604
    if Q.dr_0 < 0.064:
        z += 50.0 * Q.dr_0 - 3.2
    if Q.e3 < 0.00018:
        z += -12400.0 * Q.e3 + 2.232
    if Q.lam1 < 0.00085:
        z += -4719.0 * Q.lam1 + 5.3826
    if 0.00085 <= Q.lam1 < 0.0029:
        z += -669.0 * Q.lam1 + 1.9401
    if Q.n_pt_above_50 >= 6.0:
        z += -0.942 * Q.n_pt_above_50 + 5.652
    if Q.sum_pt < 740.0:
        z += 0.00491 * Q.sum_pt - 3.6334
    if Q.z_7 < 0.028:
        z += -283.0 * Q.z_7 + 7.924
    if Q.LHA < 0.21 and Q.log_sum_pt < 6.8:
        z += -88.5 * (0.21 - Q.LHA) * (6.8 - Q.log_sum_pt)
    if Q.LHA < 0.22 and Q.z_6 < 0.09:
        z += -524.0 * (0.22 - Q.LHA) * (0.09 - Q.z_6)
    if Q.e2 < 0.036 and Q.log_sum_pt > 6.9:
        z += -2020.0 * (0.036 - Q.e2) * (Q.log_sum_pt - 6.9)
    if Q.e2 < 0.038 and Q.z_7 < 0.073:
        z += 1490.0 * (0.038 - Q.e2) * (0.073 - Q.z_7)
    if Q.girth < 0.019 and Q.sum_pt_top5 > 700.0:
        z += 1.62 * (0.019 - Q.girth) * (Q.sum_pt_top5 - 700.0)
    if Q.lam1 < 0.0024 and Q.centroid_offset < 0.019:
        z += 113000.0 * (0.0024 - Q.lam1) * (0.019 - Q.centroid_offset)
    if Q.log_sum_pt > 6.8 and Q.e3 < 5.7e-05:
        z += -407000.0 * (Q.log_sum_pt - 6.8) * (5.7e-05 - Q.e3)
    return max(0.0, z)


def neuron_6(Q):
    z = -1.39
    if Q.C2 < 0.036:
        z += -79.3 * Q.C2 + 2.8548
    if Q.LHA >= 0.25:
        z += 45.7 * Q.LHA - 11.425
    if Q.girth >= 0.092:
        z += -238.0 * Q.girth + 21.896
    if Q.lam2 < 0.0011:
        z += 3270.0 * Q.lam2 - 3.597
    if Q.log_sum_pt < 6.4:
        z += 9.54 * Q.log_sum_pt - 61.056
    if Q.max_dr < 0.18:
        z += 27.1 * Q.max_dr - 4.878
    if Q.pt_6 < 39.0:
        z += 0.332 * Q.pt_6 - 12.948
    if Q.z_6 < 0.044:
        z += -275.0 * Q.z_6 + 12.1
    if Q.centroid_offset > 0.01 and Q.N2 > 0.19:
        z += 502.0 * (Q.centroid_offset - 0.01) * (Q.N2 - 0.19)
    if Q.centroid_offset > 0.0056 and Q.mean_phi2 < 0.017:
        z += 3970.0 * (Q.centroid_offset - 0.0056) * (0.017 - Q.mean_phi2)
    if Q.centroid_offset > 0.018 and Q.n_dr_0_0p05 < 5.8:
        z += -12.8 * (Q.centroid_offset - 0.018) * (5.8 - Q.n_dr_0_0p05)
    if Q.centroid_offset > 0.0067 and Q.sj3_dr_min < 0.22:
        z += 580.0 * (Q.centroid_offset - 0.0067) * (0.22 - Q.sj3_dr_min)
    if Q.log_sum_pt < 6.4 and Q.pt_7 < 39.0:
        z += 1.55 * (6.4 - Q.log_sum_pt) * (39.0 - Q.pt_7)
    if Q.pt_6 < 39.0 and Q.sum_pt_top3 < 720.0:
        z += 0.00168 * (39.0 - Q.pt_6) * (720.0 - Q.sum_pt_top3)
    if Q.pt_6 < 40.0 and Q.z_top3_slots < 0.79:
        z += -1.67 * (40.0 - Q.pt_6) * (0.79 - Q.z_top3_slots)
    if Q.sj2_dr > 0.16 and Q.sj2_zsoft > 0.052:
        z += 125.0 * (Q.sj2_dr - 0.16) * (Q.sj2_zsoft - 0.052)
    if Q.sj3_dr_max > 0.18 and Q.D2_b2 < 1.8:
        z += 20.3 * (Q.sj3_dr_max - 0.18) * (1.8 - Q.D2_b2)
    return max(0.0, z)


def neuron_7(Q):
    z = 2.32
    if Q.LHA >= 0.34:
        z += -231.0 * Q.LHA + 78.54
    if Q.centroid_offset < 0.029:
        z += -119.0 * Q.centroid_offset + 3.451
    if Q.e3 < 7.6e-05:
        z += 25800.0 * Q.e3 - 1.9608
    if Q.girth >= 0.1:
        z += 211.0 * Q.girth - 21.1
    if Q.lam1 < 0.0032:
        z += 2330.0 * Q.lam1 - 9.19
    if 0.0032 <= Q.lam1 < 0.0049:
        z += 1020.0 * Q.lam1 - 4.998
    if Q.sj3_dr_max < 0.15:
        z += -20.7 * Q.sj3_dr_max + 3.105
    if Q.tau1 < 0.06:
        z += -66.3 * Q.tau1 + 5.6355
    if 0.06 <= Q.tau1 < 0.085:
        z += -41.4 * Q.tau1 + 4.1415
    if Q.tau1 >= 0.085:
        z += 24.9 * Q.tau1 - 1.494
    if Q.z_dr_0p1_0p2 < 0.16:
        z += -7.66 * Q.z_dr_0p1_0p2 + 1.2256
    if Q.centroid_offset < 0.028 and Q.n_dr_0p2_0p4 < 1.0:
        z += -46.2 * (0.028 - Q.centroid_offset) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.e3 < 8.2e-05 and Q.pt_7 < 43.0:
        z += -1160.0 * (8.2e-05 - Q.e3) * (43.0 - Q.pt_7)
    if Q.girth > 0.047 and Q.D2 < 2.2:
        z += 45.1 * (Q.girth - 0.047) * (2.2 - Q.D2)
    if Q.lam1 > 0.0025 and Q.D2 > 0.31:
        z += -305.0 * (Q.lam1 - 0.0025) * (Q.D2 - 0.31)
    if Q.lam1 > 0.0022 and Q.sj3_dr_max < 0.19:
        z += -14100.0 * (Q.lam1 - 0.0022) * (0.19 - Q.sj3_dr_max)
    if Q.planar_flow < 0.17 and Q.lam1 > 0.0081:
        z += -7000.0 * (0.17 - Q.planar_flow) * (Q.lam1 - 0.0081)
    if Q.planar_flow < 0.18 and Q.log_sum_pt > 6.4:
        z += 29.2 * (0.18 - Q.planar_flow) * (Q.log_sum_pt - 6.4)
    if Q.z_dr_0p1_0p2 < 0.16 and Q.centroid_offset < 0.028:
        z += -526.0 * (0.16 - Q.z_dr_0p1_0p2) * (0.028 - Q.centroid_offset)
    return max(0.0, z)


def neuron_8(Q):
    z = -1.22
    if Q.centroid_offset >= 0.006:
        z += -129.0 * Q.centroid_offset + 0.774
    if Q.girth < 0.035:
        z += 122.0 * Q.girth - 4.27
    if Q.lam2 < 9.6e-05:
        z += -18200.0 * Q.lam2 + 1.7472
    if Q.log_sum_pt >= 6.7:
        z += -9.03 * Q.log_sum_pt + 60.501
    if Q.lam1 < 0.0072 and Q.centroid_offset < 0.024:
        z += 21400.0 * (0.0072 - Q.lam1) * (0.024 - Q.centroid_offset)
    if Q.lam1 < 0.0057 and Q.z_dr_0p2_0p4 < 0.053:
        z += 9590.0 * (0.0057 - Q.lam1) * (0.053 - Q.z_dr_0p2_0p4)
    if Q.sj3_dr_max < 0.11 and Q.centroid_offset > 0.016:
        z += -10300.0 * (0.11 - Q.sj3_dr_max) * (Q.centroid_offset - 0.016)
    if Q.sj3_dr_max < 0.2 and Q.centroid_offset > 0.0011:
        z += 1010.0 * (0.2 - Q.sj3_dr_max) * (Q.centroid_offset - 0.0011)
    if Q.sj3_dr_max < 0.14 and Q.girth2_top3 < 0.016:
        z += -1400.0 * (0.14 - Q.sj3_dr_max) * (0.016 - Q.girth2_top3)
    return max(0.0, z)


def neuron_9(Q):
    z = -5.85
    if Q.centroid_offset < 0.034:
        z += -88.9 * Q.centroid_offset + 3.0226
    if Q.girth < 0.067:
        z += 65.2 * Q.girth - 4.3684
    if Q.lam2 < 0.00019:
        z += -7580.0 * Q.lam2 + 1.4402
    if Q.sj3_dr_max < 0.13:
        z += 18.3 * Q.sj3_dr_max - 2.379
    if Q.sj3_dr_min >= 0.12:
        z += 33.9 * Q.sj3_dr_min - 4.068
    if Q.sum_pt < 930.0:
        z += -0.0111 * Q.sum_pt + 10.323
    if Q.centroid_offset < 0.024 and Q.lam1 < 0.0072:
        z += 55600.0 * (0.024 - Q.centroid_offset) * (0.0072 - Q.lam1)
    if Q.lam1 < 0.0057 and Q.lam2 < 0.0013:
        z += 1010000.0 * (0.0057 - Q.lam1) * (0.0013 - Q.lam2)
    return max(0.0, z)


def neuron_10(Q):
    z = 7.1
    if Q.girth2_top3 < 0.0029:
        z += -523.0 * Q.girth2_top3 + 1.5167
    if Q.lam1 < 0.0036:
        z += 1221.0 * Q.lam1 - 9.2732
    if 0.0036 <= Q.lam1 < 0.017:
        z += 364.0 * Q.lam1 - 6.188
    if Q.lam1 >= 0.017:
        z += 335.0 * Q.lam1 - 5.695
    if 0.00028 <= Q.lam2 < 0.0034:
        z += 675.0 * Q.lam2 - 0.189
    if Q.lam2 >= 0.0034:
        z += -253.0 * Q.lam2 + 2.9662
    if Q.sj3_dr_min >= 0.12:
        z += 18.2 * Q.sj3_dr_min - 2.184
    if Q.LHA > 0.31 and Q.planar_flow < 0.69:
        z += -56.3 * (Q.LHA - 0.31) * (0.69 - Q.planar_flow)
    if Q.lam2 > 0.00025 and Q.eccentricity > 0.47:
        z += -1630.0 * (Q.lam2 - 0.00025) * (Q.eccentricity - 0.47)
    if Q.sj2_dr > 0.18 and Q.planar_flow < 0.74:
        z += -37.1 * (Q.sj2_dr - 0.18) * (0.74 - Q.planar_flow)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.998
    if Q.centroid_offset < 0.019:
        z += -115.0 * Q.centroid_offset + 2.185
    if Q.girth < 0.071:
        z += 43.9 * Q.girth - 3.1169
    if Q.lam1 < 0.012:
        z += -570.0 * Q.lam1 + 6.84
    if Q.planar_flow < 0.31:
        z += -5.8 * Q.planar_flow + 1.798
    if Q.pt_7 < 46.0:
        z += 0.0578 * Q.pt_7 - 2.6588
    if Q.sj2_dr >= 0.12:
        z += 24.9 * Q.sj2_dr - 2.988
    if Q.sj3_dr_max >= 0.17:
        z += -45.1 * Q.sj3_dr_max + 7.667
    if Q.lam1 < 0.0035 and Q.centroid_offset < 0.027:
        z += -47400.0 * (0.0035 - Q.lam1) * (0.027 - Q.centroid_offset)
    if Q.lam1 < 0.01 and Q.centroid_offset > 0.016:
        z += -17700.0 * (0.01 - Q.lam1) * (Q.centroid_offset - 0.016)
    return max(0.0, z)


def neuron_12(Q):
    z = -2.24
    if Q.e2 >= 0.065:
        z += -66.3 * Q.e2 + 4.3095
    if Q.girth >= 0.12:
        z += 98.0 * Q.girth - 11.76
    if Q.sd_rg >= 0.31:
        z += 52.4 * Q.sd_rg - 16.244
    if Q.sd_rg > 0.31 and Q.log_sum_pt < 6.9:
        z += -62.5 * (Q.sd_rg - 0.31) * (6.9 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.587
    if Q.centroid_offset < 0.018:
        z += 89.8 * Q.centroid_offset - 1.6164
    if Q.girth < 0.15:
        z += -65.6 * Q.girth + 9.84
    if Q.lam1 < 0.016:
        z += -256.0 * Q.lam1 + 4.096
    if Q.pt_7 >= 35.0:
        z += -0.0894 * Q.pt_7 + 3.129
    if Q.sj3_dr_max >= 0.23:
        z += -12.1 * Q.sj3_dr_max + 2.783
    if Q.sj3_dr_min < 0.13:
        z += 20.5 * Q.sj3_dr_min - 2.665
    if Q.tau1 < 0.11:
        z += 13.4 * Q.tau1 - 1.474
    if Q.z_7 >= 0.045:
        z += 61.4 * Q.z_7 - 2.763
    if Q.girth < 0.15 and Q.log_sum_pt < 6.7:
        z += -102.0 * (0.15 - Q.girth) * (6.7 - Q.log_sum_pt)
    if Q.girth < 0.15 and Q.tau2 < 0.036:
        z += -628.0 * (0.15 - Q.girth) * (0.036 - Q.tau2)
    if Q.lam1 < 0.016 and Q.pt_7 < 26.0:
        z += -23.6 * (0.016 - Q.lam1) * (26.0 - Q.pt_7)
    if Q.sum_pt_top5 > 690.0 and Q.pt_7 < 39.0:
        z += 0.000747 * (Q.sum_pt_top5 - 690.0) * (39.0 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = -4.29
    if Q.centroid_offset >= 0.05:
        z += -1200.0 * Q.centroid_offset + 60.0
    if Q.e2 < 0.041:
        z += -130.0 * Q.e2 + 5.33
    if Q.e3 < 7.9e-05:
        z += 17100.0 * Q.e3 - 1.3509
    if 0.027 <= Q.girth < 0.088:
        z += 129.0 * Q.girth - 3.483
    if Q.girth >= 0.088:
        z += -199.0 * Q.girth + 25.381
    if Q.lam1 < 0.0065:
        z += 1150.0 * Q.lam1 - 7.475
    if Q.lam2 < 0.0034:
        z += 1830.0 * Q.lam2 - 6.222
    if Q.sd_rg < 0.16:
        z += -12.5 * Q.sd_rg + 3.0
    if 0.16 <= Q.sd_rg < 0.19:
        z += 15.6 * Q.sd_rg - 1.496
    if 0.19 <= Q.sd_rg < 0.24:
        z += -57.1 * Q.sd_rg + 12.317
    if Q.sd_rg >= 0.24:
        z += -44.6 * Q.sd_rg + 9.317
    if Q.tau1 < 0.096:
        z += -94.0 * Q.tau1 + 9.992
    if 0.096 <= Q.tau1 < 0.14:
        z += -22.0 * Q.tau1 + 3.08
    if Q.z_dr_0p05_0p1 >= 0.17:
        z += -3.19 * Q.z_dr_0p05_0p1 + 0.5423
    if Q.lam1 < 0.016 and Q.centroid_offset > 0.027:
        z += -20500.0 * (0.016 - Q.lam1) * (Q.centroid_offset - 0.027)
    if Q.lam2 < 0.0034 and Q.LHA > 0.18:
        z += 4080.0 * (0.0034 - Q.lam2) * (Q.LHA - 0.18)
    if Q.lam2 < 0.0034 and Q.sd_rg > 0.16:
        z += 14300.0 * (0.0034 - Q.lam2) * (Q.sd_rg - 0.16)
    if Q.planar_flow < 0.11 and Q.lam1 < 0.0059:
        z += 13100.0 * (0.11 - Q.planar_flow) * (0.0059 - Q.lam1)
    if Q.planar_flow < 0.11 and Q.lam1 < 0.0073:
        z += -18500.0 * (0.11 - Q.planar_flow) * (0.0073 - Q.lam1)
    if Q.planar_flow < 0.11 and Q.lam1 < 0.016:
        z += 1670.0 * (0.11 - Q.planar_flow) * (0.016 - Q.lam1)
    if Q.planar_flow < 0.11 and Q.sj2_zsoft < 0.28:
        z += 85.0 * (0.11 - Q.planar_flow) * (0.28 - Q.sj2_zsoft)
    if Q.z_dr_0p05_0p1 > 0.16 and Q.log_sum_pt < 6.7:
        z += -4.61 * (Q.z_dr_0p05_0p1 - 0.16) * (6.7 - Q.log_sum_pt)
    if Q.z_dr_0p05_0p1 > 0.16 and Q.n_dr_0p2_0p4 < 1.0:
        z += 4.94 * (Q.z_dr_0p05_0p1 - 0.16) * (1.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_15(Q):
    z = -1.81
    if Q.eccentricity >= 0.94:
        z += -29.8 * Q.eccentricity + 28.012
    if Q.girth < 0.033:
        z += 260.9 * Q.girth - 12.2919
    if 0.033 <= Q.girth < 0.071:
        z += 96.9 * Q.girth - 6.8799
    if Q.girth2_top2 < 0.0081:
        z += -195.0 * Q.girth2_top2 + 1.5795
    if Q.lam1 < 0.0071:
        z += 733.0 * Q.lam1 - 5.2043
    if Q.lam2 < 0.0033:
        z += -1100.0 * Q.lam2 + 3.63
    if Q.tau1 < 0.053:
        z += -118.0 * Q.tau1 + 6.254
    if Q.tau1 >= 0.14:
        z += -29.7 * Q.tau1 + 4.158
    if Q.N2 < 0.16 and Q.LHA > 0.38:
        z += -1880.0 * (0.16 - Q.N2) * (Q.LHA - 0.38)
    if Q.N2 < 0.21 and Q.LHA > 0.28:
        z += 338.0 * (0.21 - Q.N2) * (Q.LHA - 0.28)
    if Q.N2 < 0.22 and Q.lam1 > 0.0081:
        z += -5800.0 * (0.22 - Q.N2) * (Q.lam1 - 0.0081)
    if Q.N2 < 0.23 and Q.sj2_dr < 0.19:
        z += -337.0 * (0.23 - Q.N2) * (0.19 - Q.sj2_dr)
    if Q.eccentricity > 0.94 and Q.sum_pt_top5 > 370.0:
        z += 0.122 * (Q.eccentricity - 0.94) * (Q.sum_pt_top5 - 370.0)
    if Q.girth2_top3 < 0.0022 and Q.planar_flow < 0.14:
        z += -5600.0 * (0.0022 - Q.girth2_top3) * (0.14 - Q.planar_flow)
    if Q.lam1 < 0.0065 and Q.D2 < 0.85:
        z += 3900.0 * (0.0065 - Q.lam1) * (0.85 - Q.D2)
    if Q.lam1 < 0.0087 and Q.D2 < 0.86:
        z += -1840.0 * (0.0087 - Q.lam1) * (0.86 - Q.D2)
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
