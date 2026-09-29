"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; no mass observables), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.8% (the network: 65.8%); same class as the network for 87.6% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr12                   ΔR between particles 1 and 2
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
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
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        max_dr=max(dr[i] for i in real),
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_5=z[5] * dr[5],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = 7.295449
    if Q.planar_flow < 0.1484197:
        z += -6.290462 * Q.planar_flow + 0.9336286
    if Q.lam1_plus_lam2 < 0.003562611:
        z += 374.5927 * Q.lam1_plus_lam2 + 3.186036
    if 0.003562611 <= Q.lam1_plus_lam2 < 0.004372139:
        z += -569.2043 * Q.lam1_plus_lam2 + 6.548417
    if 0.004372139 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -942.8392 * Q.lam1_plus_lam2 + 8.182001
    if Q.sum_z_dr2 < 0.005590289:
        z += 96.99162 * Q.sum_z_dr2 + 3.094254
    if 0.005590289 <= Q.sum_z_dr2 < 0.01323868:
        z += -406.3626 * Q.sum_z_dr2 + 5.908149
    if 0.01323868 <= Q.sum_z_dr2 < 0.01882765:
        z += -94.55169 * Q.sum_z_dr2 + 1.780186
    if Q.sum_pt < 739.5:
        z += 0.01280497 * Q.sum_pt - 9.877477
    if 739.5 <= Q.sum_pt < 788.4484:
        z += 0.008339444 * Q.sum_pt - 6.575222
    if Q.sum_z_dr2_top3 < 0.007929074:
        z += 148.3581 * Q.sum_z_dr2_top3 - 1.176343
    if Q.tau1 < 0.06345984:
        z += -58.64101 * Q.tau1 + 3.721349
    z += -1.39177 * Q.log_sum_pt
    if Q.e3 < 8.147744e-05:
        z += -17378.05 * Q.e3 + 1.415919
    if Q.tau2 < 0.01357153:
        z += 180.5396 * Q.tau2 - 2.450199
    if Q.sd_rg < 0.06545715:
        z += -1.786217 * Q.sd_rg - 0.2188836
    if 0.06545715 <= Q.sd_rg < 0.1116471:
        z += -19.12003 * Q.sd_rg + 0.9157381
    if 0.1116471 <= Q.sd_rg < 0.1773029:
        z += 18.56584 * Q.sd_rg - 3.291778
    if Q.centroid_offset >= 0.04990367:
        z += -440.6739 * Q.centroid_offset + 21.99124
    if Q.lam1 < 0.008375572:
        z += 324.3455 * Q.lam1 - 2.716579
    if Q.eccentricity >= 0.9884745:
        z += 98.3032 * Q.eccentricity - 97.17021
    if Q.sum_pt_top5 < 631.275:
        z += -0.01106508 * Q.sum_pt_top5 + 6.985105
    if Q.n_dr_0_0p05 >= 5.0:
        z += 0.3736244 * Q.n_dr_0_0p05 - 1.868122
    if Q.sum_z_dr < 0.08723651:
        z += 52.49235 * Q.sum_z_dr - 4.579249
    if Q.sum_zz_dr2 < 0.00718279:
        z += 389.3177 * Q.sum_zz_dr2 - 2.796388
    if Q.sum_pt < 788.4484 and Q.M2 > 0.02563286:
        z += 0.04098614 * (788.4484 - Q.sum_pt) * (Q.M2 - 0.02563286)
    if Q.sum_z_dr2 < 0.01882765 and Q.M3 < 0.09029177:
        z += 2151.545 * (0.01882765 - Q.sum_z_dr2) * (0.09029177 - Q.M3)
    if Q.lam1_plus_lam2 < 0.004372139 and Q.dr0_6 > 0.1755206:
        z += 14340.65 * (0.004372139 - Q.lam1_plus_lam2) * (Q.dr0_6 - 0.1755206)
    if Q.e3 < 8.147744e-05 and Q.dr_6 > 0.1695089:
        z += 500445.2 * (8.147744e-05 - Q.e3) * (Q.dr_6 - 0.1695089)
    if Q.planar_flow < 0.1484197 and Q.D2_b2 < 1.129616:
        z += 9.382482 * (0.1484197 - Q.planar_flow) * (1.129616 - Q.D2_b2)
    if Q.sum_pt < 788.4484 and Q.D2_b2 < 0.380911:
        z += -0.02016008 * (788.4484 - Q.sum_pt) * (0.380911 - Q.D2_b2)
    if Q.sum_z_dr2 < 0.01882765 and Q.centroid_offset > 0.009480685:
        z += -4614.381 * (0.01882765 - Q.sum_z_dr2) * (Q.centroid_offset - 0.009480685)
    if Q.D2_b2 < 0.380911 and Q.pt_5 > 24.57812:
        z += -0.06040816 * (0.380911 - Q.D2_b2) * (Q.pt_5 - 24.57812)
    if Q.sum_z_dr < 0.01517359 and Q.dr0_6 > 0.1979161:
        z += -71344.4 * (0.01517359 - Q.sum_z_dr) * (Q.dr0_6 - 0.1979161)
    if Q.tau1 < 0.06345984 and Q.dr0_6 > 0.1755206:
        z += -1356.6 * (0.06345984 - Q.tau1) * (Q.dr0_6 - 0.1755206)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.5778859
    if Q.pt_7 >= 34.53125:
        z += 0.1168804 * Q.pt_7 - 4.036027
    if 6.377723 <= Q.log_sum_pt < 6.572938:
        z += 4.834097 * Q.log_sum_pt - 30.83053
    if Q.log_sum_pt >= 6.572938:
        z += 16.32537 * Q.log_sum_pt - 106.362
    if Q.sum_zz_dr2 < 0.005834489:
        z += -593.1165 * Q.sum_zz_dr2 + 3.460532
    if Q.z_7 < 0.05240025:
        z += 139.2906 * Q.z_7 - 7.818258
    if 0.05240025 <= Q.z_7 < 0.06164517:
        z += 56.18194 * Q.z_7 - 3.463345
    if Q.lam1_plus_lam2 < 0.008678045:
        z += 445.8694 * Q.lam1_plus_lam2 - 3.869275
    if Q.tau1 < 0.07283629:
        z += -5.454029 * Q.tau1 + 1.037767
    if 0.07283629 <= Q.tau1 < 0.1027642:
        z += -21.40192 * Q.tau1 + 2.199352
    if Q.lam1 < 0.002464291:
        z += -305.1952 * Q.lam1 + 0.7520899
    if Q.zdr_0 < 0.0211821:
        z += 55.22369 * Q.zdr_0 - 1.169754
    if Q.sum_z_dr2 < 0.00609665:
        z += 450.272 * Q.sum_z_dr2 - 2.745151
    if Q.sum_z_dr < 0.1019409:
        z += 22.09417 * Q.sum_z_dr - 2.2523
    if Q.pt_7 > 34.53125 and Q.tau2 < 0.01713288:
        z += -5.711835 * (Q.pt_7 - 34.53125) * (0.01713288 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.e3 < 0.0001869378:
        z += 190384.4 * (0.06164517 - Q.z_7) * (0.0001869378 - Q.e3)
    if Q.sj2_dr > 0.1872617 and Q.sj3_dr_min < 0.2089872:
        z += 50.27641 * (Q.sj2_dr - 0.1872617) * (0.2089872 - Q.sj3_dr_min)
    if Q.z_7 < 0.06164517 and Q.sum_z_dr2_top2 < 0.01403324:
        z += 3338.645 * (0.06164517 - Q.z_7) * (0.01403324 - Q.sum_z_dr2_top2)
    if Q.log_sum_pt > 6.377723 and Q.dr_0 < 0.08082334:
        z += -72.23889 * (Q.log_sum_pt - 6.377723) * (0.08082334 - Q.dr_0)
    if Q.lam1 < 0.008375572 and Q.dr_0 < 0.1442881:
        z += 2649.989 * (0.008375572 - Q.lam1) * (0.1442881 - Q.dr_0)
    if Q.log_sum_pt > 6.572938 and Q.sum_z_dr2_top3 < 0.007929074:
        z += -410.6432 * (Q.log_sum_pt - 6.572938) * (0.007929074 - Q.sum_z_dr2_top3)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.6313048
    if Q.pt_7 < 53.4375:
        z += 0.1541831 * Q.pt_7 - 4.743323
    if Q.pt_7 >= 53.4375:
        z += 0.06541922 * Q.pt_7
    if Q.log_sum_pt < 6.605974:
        z += -5.886923 * Q.log_sum_pt + 39.69118
    if 6.605974 <= Q.log_sum_pt < 6.701242:
        z += -8.421732 * Q.log_sum_pt + 56.43607
    if Q.log_sum_pt >= 6.896095:
        z += 16.76466 * Q.log_sum_pt - 115.6107
    if 0.02320757 <= Q.z_7 < 0.02807091:
        z += -57.01788 * Q.z_7 + 1.323246
    if Q.z_7 >= 0.02807091:
        z += -79.14788 * Q.z_7 + 1.944456
    if Q.sum_z_dr < 0.007673833:
        z += 413.4086 * Q.sum_z_dr - 3.172429
    if Q.LHA >= 0.111565:
        z += -6.167074 * Q.LHA + 0.6880299
    if Q.e2 < 0.04447357:
        z += 40.67015 * Q.e2 - 1.808747
    if Q.sum_pt < 615.875:
        z += -0.008981846 * Q.sum_pt + 5.531695
    if 937.0312 <= Q.sum_pt < 988.4078:
        z += 0.009129544 * Q.sum_pt - 8.554668
    if Q.sum_pt >= 988.4078:
        z += -0.02415004 * Q.sum_pt + 24.33914
    if Q.C2 < 0.0423228:
        z += -71.94674 * Q.C2 + 3.044988
    if Q.lam1_plus_lam2 < 0.01323868:
        z += -140.2014 * Q.lam1_plus_lam2 + 1.85608
    if Q.sum_zz_dr2 < 0.0030133:
        z += 220.5433 * Q.sum_zz_dr2 - 0.6645632
    if Q.N2 >= 0.071455:
        z += 4.913432 * Q.N2 - 0.3510893
    if Q.zdr_0 < 0.02383244:
        z += -50.11039 * Q.zdr_0 + 1.194253
    if Q.sj3_dr_min < 0.1278212:
        z += -9.436476 * Q.sj3_dr_min + 1.206181
    if Q.lam1 < 0.00733008 and Q.max_dr < 0.2507612:
        z += 1086.224 * (0.00733008 - Q.lam1) * (0.2507612 - Q.max_dr)
    if Q.sum_pt_top5 > 791.125 and Q.D2_b2 < 1.129616:
        z += 0.02151631 * (Q.sum_pt_top5 - 791.125) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 1.59265:
        z += -16.24811 * (Q.log_sum_pt - 6.896095) * (1.59265 - Q.D2_b2)
    if Q.log_sum_pt < 6.605974 and Q.D2_b2 < 1.129616:
        z += 6.054007 * (6.605974 - Q.log_sum_pt) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt < 6.701242 and Q.D2_b2 < 1.129616:
        z += -5.07238 * (6.701242 - Q.log_sum_pt) * (1.129616 - Q.D2_b2)
    if Q.centroid_offset < 0.005576073 and Q.C3 < 0.01685631:
        z += -24234.05 * (0.005576073 - Q.centroid_offset) * (0.01685631 - Q.C3)
    if Q.C2 < 0.0423228 and Q.C2_b2 < 0.02415398:
        z += -3879.649 * (0.0423228 - Q.C2) * (0.02415398 - Q.C2_b2)
    return max(0.0, z)


def neuron_3(Q):
    z = -1.790616
    if 0.06663269 <= Q.sum_z_dr < 0.08723651:
        z += 99.32594 * Q.sum_z_dr - 6.618355
    if Q.sum_z_dr >= 0.08723651:
        z += 146.5794 * Q.sum_z_dr - 10.74058
    if 0.1682655 <= Q.sj2_dr < 0.1778793:
        z += 15.20021 * Q.sj2_dr - 2.557672
    if 0.1778793 <= Q.sj2_dr < 0.2687922:
        z += 46.38777 * Q.sj2_dr - 8.105293
    if 0.2687922 <= Q.sj2_dr < 0.3003793:
        z += 35.15695 * Q.sj2_dr - 5.086537
    if Q.sj2_dr >= 0.3003793:
        z += 71.86735 * Q.sj2_dr - 16.11358
    if 0.00609665 <= Q.lam1_plus_lam2 < 0.01323868:
        z += 539.2403 * Q.lam1_plus_lam2 - 3.287559
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += 249.4443 * Q.lam1_plus_lam2 + 0.5489551
    if 0.006390125 <= Q.sum_zz_dr2 < 0.00718279:
        z += -622.9854 * Q.sum_zz_dr2 + 3.980954
    if 0.00718279 <= Q.sum_zz_dr2 < 0.01165737:
        z += 101.3175 * Q.sum_zz_dr2 - 1.221561
    if Q.sum_zz_dr2 >= 0.01165737:
        z += 41.59936 * Q.sum_zz_dr2 - 0.5254042
    if Q.sum_z_dr2 >= 0.008678045:
        z += -1007.402 * Q.sum_z_dr2 + 8.742281
    if 0.0811449 <= Q.tau1 < 0.1136369:
        z += -49.24825 * Q.tau1 + 3.996244
    if Q.tau1 >= 0.1136369:
        z += 25.86392 * Q.tau1 - 4.539269
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += 449.2146 * Q.lam1 - 3.762429
    if Q.lam1 >= 0.01200373:
        z += 268.8946 * Q.lam1 - 1.597917
    if Q.lam2 >= 0.000537286:
        z += 418.9359 * Q.lam2 - 0.2250884
    if Q.sj2_dr > 0.1682655 and Q.sum_pt < 988.4078:
        z += -0.02377606 * (Q.sj2_dr - 0.1682655) * (988.4078 - Q.sum_pt)
    if Q.sj2_dr > 0.1682655 and Q.tau2 < 0.06297984:
        z += 576.4404 * (Q.sj2_dr - 0.1682655) * (0.06297984 - Q.tau2)
    if Q.lam1_plus_lam2 > 0.00609665 and Q.n_pt_above_50 > 3.0:
        z += -26.16523 * (Q.lam1_plus_lam2 - 0.00609665) * (Q.n_pt_above_50 - 3.0)
    if Q.sj2_dr > 0.1682655 and Q.n_dr_0p05_0p1 < 7.0:
        z += -5.690283 * (Q.sj2_dr - 0.1682655) * (7.0 - Q.n_dr_0p05_0p1)
    if Q.lam1_plus_lam2 > 0.00609665 and Q.log_sum_pt > 6.192222:
        z += -431.6542 * (Q.lam1_plus_lam2 - 0.00609665) * (Q.log_sum_pt - 6.192222)
    if Q.sj2_dr > 0.3003793 and Q.z_dr_0p05_0p1 > 0.08878489:
        z += -146.1689 * (Q.sj2_dr - 0.3003793) * (Q.z_dr_0p05_0p1 - 0.08878489)
    if Q.sj2_dr > 0.1778793 and Q.eccentricity > 0.9458207:
        z += 673.7551 * (Q.sj2_dr - 0.1778793) * (Q.eccentricity - 0.9458207)
    if Q.sj2_dr > 0.1682655 and Q.eccentricity > 0.9458207:
        z += -597.6909 * (Q.sj2_dr - 0.1682655) * (Q.eccentricity - 0.9458207)
    if Q.sj2_dr > 0.1778793 and Q.sum_z_dr2_top2 < 0.001864148:
        z += -33850.32 * (Q.sj2_dr - 0.1778793) * (0.001864148 - Q.sum_z_dr2_top2)
    if Q.sj2_dr > 0.3003793 and Q.D2_b2 < 0.01618699:
        z += -10202.41 * (Q.sj2_dr - 0.3003793) * (0.01618699 - Q.D2_b2)
    return max(0.0, z)


def neuron_4(Q):
    z = -0.5965389
    if Q.N2 < 0.2233283:
        z += -30.24196 * Q.N2 + 6.753885
    if Q.sum_z_dr2 < 0.002635418:
        z += -298.8339 * Q.sum_z_dr2 - 5.970608
    if 0.002635418 <= Q.sum_z_dr2 < 0.006679471:
        z += 1270.05 * Q.sum_z_dr2 - 10.10527
    if 0.006679471 <= Q.sum_z_dr2 < 0.008678045:
        z += 811.5841 * Q.sum_z_dr2 - 7.042963
    if Q.e3 < 0.0001869378:
        z += -24210.03 * Q.e3 + 4.52577
    if Q.lam1 < 0.002464291:
        z += 187.9084 * Q.lam1 + 0.69479
    if 0.002464291 <= Q.lam1 < 0.00483998:
        z += -487.3748 * Q.lam1 + 2.358885
    if Q.sj3_dr_min < 0.2089872:
        z += -18.30394 * Q.sj3_dr_min + 3.82529
    if Q.lam2 < 0.001130645:
        z += 4112.223 * Q.lam2 - 4.649463
    if 0.1789613 <= Q.sj3_dr_max < 0.213399:
        z += 12.45677 * Q.sj3_dr_max - 2.22928
    if 0.213399 <= Q.sj3_dr_max < 0.3456459:
        z += -10.35515 * Q.sj3_dr_max + 2.63876
    if Q.sj3_dr_max >= 0.3456459:
        z += 14.96718 * Q.sj3_dr_max - 6.113801
    if Q.sum_z_dr < 0.07608178:
        z += -21.49528 * Q.sum_z_dr + 1.635399
    if Q.e2 < 0.04447357:
        z += 51.06131 * Q.e2 - 2.270879
    if Q.sum_zz_dr2 < 0.01165737:
        z += -280.3064 * Q.sum_zz_dr2 + 3.267636
    if Q.C2_b2 < 0.009032972:
        z += -204.0416 * Q.C2_b2 + 1.843102
    if Q.C2 >= 0.02384388:
        z += -15.40846 * Q.C2 + 0.3673975
    if Q.N2 < 0.2233283 and Q.planar_flow < 0.4007947:
        z += -71.74352 * (0.2233283 - Q.N2) * (0.4007947 - Q.planar_flow)
    if Q.e3 < 0.0001869378 and Q.sum_pt < 813.4156:
        z += -23.90663 * (0.0001869378 - Q.e3) * (813.4156 - Q.sum_pt)
    if Q.sum_z_dr2_top2 < 0.006299534 and Q.centroid_offset > 0.01096064:
        z += 26577.79 * (0.006299534 - Q.sum_z_dr2_top2) * (Q.centroid_offset - 0.01096064)
    if Q.sj3_dr_max > 0.3456459 and Q.D2_b2 < 1.129616:
        z += -59.37822 * (Q.sj3_dr_max - 0.3456459) * (1.129616 - Q.D2_b2)
    if Q.sum_zz_dr2 < 0.01165737 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -2905.921 * (0.01165737 - Q.sum_zz_dr2) * (0.05643852 - Q.z_dr_0p2_0p4)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.2543399
    if Q.LHA < 0.2160559:
        z += -32.79946 * Q.LHA + 7.086516
    if Q.e2 < 0.03556091:
        z += -56.48883 * Q.e2 + 2.008794
    if Q.log_sum_pt >= 6.896095:
        z += -32.71606 * Q.log_sum_pt + 225.6131
    if Q.sum_z_dr2 < 0.0009641429:
        z += -5012.035 * Q.sum_z_dr2 + 7.350226
    if 0.0009641429 <= Q.sum_z_dr2 < 0.002635418:
        z += -1130.62 * Q.sum_z_dr2 + 3.607987
    if 0.002635418 <= Q.sum_z_dr2 < 0.005019719:
        z += -263.5282 * Q.sum_z_dr2 + 1.322837
    if Q.sum_pt < 715.4688:
        z += 0.01618676 * Q.sum_pt - 11.58112
    if Q.centroid_offset < 0.03117077:
        z += 68.56624 * Q.centroid_offset - 2.137263
    if Q.n_pt_above_50 >= 6.0:
        z += -0.5588672 * Q.n_pt_above_50 + 3.353203
    if Q.z_7 < 0.02807091:
        z += -251.7839 * Q.z_7 + 7.067805
    if Q.sum_pt_top5 < 430.75:
        z += 0.07849395 * Q.sum_pt_top5 - 33.81127
    if Q.dr_0 < 0.04649465:
        z += 81.209 * Q.dr_0 - 3.775784
    if Q.sj3_dr_max < 0.3012016:
        z += 8.397539 * Q.sj3_dr_max - 2.529352
    if Q.e3 < 0.0005116989:
        z += -3323.289 * Q.e3 + 1.700523
    if Q.z_6 < 0.02160287:
        z += -202.4065 * Q.z_6 + 5.448223
    if 0.02160287 <= Q.z_6 < 0.06081235:
        z += -27.43367 * Q.z_6 + 1.668306
    if Q.pt_7 < 20.125:
        z += 0.2091096 * Q.pt_7 - 4.20833
    if Q.pt_7 >= 34.53125:
        z += -0.03701957 * Q.pt_7 + 1.278332
    if Q.LHA < 0.2160559 and Q.z_6 < 0.09543973:
        z += -383.4601 * (0.2160559 - Q.LHA) * (0.09543973 - Q.z_6)
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -180.2635 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.e2 < 0.03556091 and Q.log_sum_pt > 6.896095:
        z += -1496.649 * (0.03556091 - Q.e2) * (Q.log_sum_pt - 6.896095)
    if Q.sum_z_dr2 < 0.002635418 and Q.centroid_offset < 0.01627885:
        z += 60815.5 * (0.002635418 - Q.sum_z_dr2) * (0.01627885 - Q.centroid_offset)
    if Q.LHA < 0.1767241 and Q.zdr_0 < 0.006292091:
        z += -2310.655 * (0.1767241 - Q.LHA) * (0.006292091 - Q.zdr_0)
    if Q.sum_pt < 715.4688 and Q.zdr_0 < 0.007798268:
        z += 4.200004 * (715.4688 - Q.sum_pt) * (0.007798268 - Q.zdr_0)
    if Q.dr_0 < 0.04649465 and Q.N3 < 2.080881:
        z += -36.9208 * (0.04649465 - Q.dr_0) * (2.080881 - Q.N3)
    if Q.centroid_offset < 0.03117077 and Q.tau3 < 0.01364517:
        z += 4100.235 * (0.03117077 - Q.centroid_offset) * (0.01364517 - Q.tau3)
    if Q.e2 < 0.03556091 and Q.planar_flow < 0.3220738:
        z += 136.1072 * (0.03556091 - Q.e2) * (0.3220738 - Q.planar_flow)
    if Q.z_7 < 0.04939969 and Q.absphi_0 < 0.04678345:
        z += 970.2171 * (0.04939969 - Q.z_7) * (0.04678345 - Q.absphi_0)
    if Q.log_sum_pt > 6.896095 and Q.mean_eta2 < 5.507482e-05:
        z += 540619.8 * (Q.log_sum_pt - 6.896095) * (5.507482e-05 - Q.mean_eta2)
    return max(0.0, z)


def neuron_6(Q):
    z = 2.036704
    if 0.00809236 <= Q.centroid_offset < 0.01627885:
        z += 94.91757 * Q.centroid_offset - 0.7681072
    if Q.centroid_offset >= 0.01627885:
        z += 180.7139 * Q.centroid_offset - 2.164772
    if Q.lam1_plus_lam2 < 0.01323868:
        z += 421.3812 * Q.lam1_plus_lam2 - 5.578529
    if Q.tau1 < 0.1136369:
        z += -42.4539 * Q.tau1 + 4.824329
    if Q.log_sum_pt < 6.327379:
        z += -14.64551 * Q.log_sum_pt + 92.66769
    if Q.tau2 >= 0.01713288:
        z += -41.12882 * Q.tau2 + 0.7046551
    if Q.lam2 < 0.001130645:
        z += 271.902 * Q.lam2 + 1.259048
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -687.7297 * Q.lam2 + 2.34405
    if Q.lam1 < 0.008375572:
        z += -665.8796 * Q.lam1 + 7.097302
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += -418.9951 * Q.lam1 + 5.029503
    if Q.z_6 < 0.02160287:
        z += 257.7704 * Q.z_6 - 5.568582
    if Q.sj3_dr_min >= 0.02022982:
        z += -13.07922 * Q.sj3_dr_min + 0.2645902
    if Q.sum_z_dr2 < 0.008678045:
        z += 1917.04 * Q.sum_z_dr2 - 16.63616
    if 0.07264452 <= Q.sj3_dr_max < 0.1789613:
        z += -12.10448 * Q.sj3_dr_max + 0.879324
    if Q.sj3_dr_max >= 0.1789613:
        z += 11.12829 * Q.sj3_dr_max - 3.278443
    if Q.sum_zz_dr2 < 0.008168571:
        z += -405.8031 * Q.sum_zz_dr2 + 3.314831
    if Q.sum_pt >= 988.4078:
        z += 0.0143338 * Q.sum_pt - 14.16764
    if Q.max_dr < 0.1598486:
        z += 30.99929 * Q.max_dr - 4.955195
    if Q.sum_z_dr < 0.02689598:
        z += -281.7702 * Q.sum_z_dr + 7.578486
    if Q.n_dr_0_0p05 < 3.0:
        z += -0.295114 * Q.n_dr_0_0p05 + 0.8853421
    if Q.centroid_offset > 0.00809236 and Q.sj2_dr < 0.1682655:
        z += 1076.214 * (Q.centroid_offset - 0.00809236) * (0.1682655 - Q.sj2_dr)
    if Q.log_sum_pt < 6.327379 and Q.z_6 < 0.08051087:
        z += 414.0089 * (6.327379 - Q.log_sum_pt) * (0.08051087 - Q.z_6)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.planar_flow < 0.3220738:
        z += -565.6522 * (0.01323868 - Q.lam1_plus_lam2) * (0.3220738 - Q.planar_flow)
    if Q.centroid_offset > 0.00809236 and Q.n_dr_0p05_0p1 < 6.0:
        z += 9.250124 * (Q.centroid_offset - 0.00809236) * (6.0 - Q.n_dr_0p05_0p1)
    if Q.log_sum_pt < 6.327379 and Q.pt_7 > 25.57812:
        z += -0.7544965 * (6.327379 - Q.log_sum_pt) * (Q.pt_7 - 25.57812)
    if Q.lam1 < 0.01200373 and Q.pt_6 < 19.46875:
        z += 19.61145 * (0.01200373 - Q.lam1) * (19.46875 - Q.pt_6)
    if Q.z_6 < 0.02160287 and Q.log_sum_pt < 6.842717:
        z += 5607.925 * (0.02160287 - Q.z_6) * (6.842717 - Q.log_sum_pt)
    if Q.log_sum_pt < 6.638339 and Q.z_7 < 0.0586137:
        z += 625.6663 * (6.638339 - Q.log_sum_pt) * (0.0586137 - Q.z_7)
    if Q.centroid_offset > 0.00809236 and Q.mean_phi2 < 0.01426135:
        z += 4107.327 * (Q.centroid_offset - 0.00809236) * (0.01426135 - Q.mean_phi2)
    if Q.centroid_offset > 0.01627885 and Q.sum_pt < 988.4078:
        z += -0.5658113 * (Q.centroid_offset - 0.01627885) * (988.4078 - Q.sum_pt)
    return max(0.0, z)


def neuron_7(Q):
    z = 9.937822
    if Q.planar_flow < 0.1950135:
        z += 10.01187 * Q.planar_flow - 1.952449
    if Q.sum_z_dr2 < 0.0005611231:
        z += -1001.99 * Q.sum_z_dr2 - 0.2587063
    if 0.0005611231 <= Q.sum_z_dr2 < 0.0009641429:
        z += 2036.987 * Q.sum_z_dr2 - 1.963947
    if 0.004372139 <= Q.sum_z_dr2 < 0.007520088:
        z += -769.1769 * Q.sum_z_dr2 + 3.362949
    if 0.007520088 <= Q.sum_z_dr2 < 0.008678045:
        z += -2296.72 * Q.sum_z_dr2 + 14.85021
    if 0.008678045 <= Q.sum_z_dr2 < 0.01323868:
        z += -1323.169 * Q.sum_z_dr2 + 6.401691
    if Q.sum_z_dr2 >= 0.01323868:
        z += 33.75275 * Q.sum_z_dr2 - 11.56216
    if Q.N2 >= 0.198361:
        z += -4.283192 * Q.N2 + 0.8496181
    if Q.tau1 < 0.05356915:
        z += -60.77407 * Q.tau1 + 3.906594
    if 0.05356915 <= Q.tau1 < 0.09538712:
        z += -21.12115 * Q.tau1 + 1.78242
    if 0.09538712 <= Q.tau1 < 0.1136369:
        z += 39.23102 * Q.tau1 - 3.9744
    if 0.1136369 <= Q.tau1 < 0.1416226:
        z += -17.28351 * Q.tau1 + 2.447736
    if Q.D2 < 2.843757:
        z += 0.3296404 * Q.D2 - 0.9374171
    if Q.sum_zz_dr2 < 0.005284669:
        z += 14.80273 * Q.sum_zz_dr2 - 3.2849
    if 0.005284669 <= Q.sum_zz_dr2 < 0.00718279:
        z += 818.1005 * Q.sum_zz_dr2 - 7.530063
    if 0.00718279 <= Q.sum_zz_dr2 < 0.008168571:
        z += 1677.674 * Q.sum_zz_dr2 - 13.7042
    if Q.lam1_plus_lam2 < 0.005590289:
        z += 1195.659 * Q.lam1_plus_lam2 - 6.684079
    if Q.sj2_dr < 0.1294903:
        z += -22.72411 * Q.sj2_dr + 2.25328
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -9.179462 * Q.sj2_dr + 0.4993793
    if 0.1591713 <= Q.sj2_dr < 0.1872617:
        z += 34.23686 * Q.sj2_dr - 6.411253
    if Q.centroid_offset >= 0.009480685:
        z += 22.09837 * Q.centroid_offset - 0.2095077
    if Q.sum_z_dr < 0.08723651:
        z += 70.08975 * Q.sum_z_dr - 6.114386
    if Q.LHA < 0.2931906:
        z += -4.644068 * Q.LHA + 1.361597
    if Q.lam2 < 0.001130645:
        z += -1221.462 * Q.lam2 + 1.381039
    if Q.e3 < 0.0001869378:
        z += 4742.694 * Q.e3 - 0.8865889
    if Q.planar_flow < 0.1950135 and Q.lam1_plus_lam2 > 0.007520088:
        z += -2937.948 * (0.1950135 - Q.planar_flow) * (Q.lam1_plus_lam2 - 0.007520088)
    if Q.planar_flow < 0.1950135 and Q.log_sum_pt > 6.423044:
        z += 50.54787 * (0.1950135 - Q.planar_flow) * (Q.log_sum_pt - 6.423044)
    if Q.planar_flow < 0.1950135 and Q.z_7 < 0.05240025:
        z += -434.4552 * (0.1950135 - Q.planar_flow) * (0.05240025 - Q.z_7)
    if Q.tau1 < 0.05356915 and Q.z_dr_0p05_0p1 > 0.6747704:
        z += -339.2826 * (0.05356915 - Q.tau1) * (Q.z_dr_0p05_0p1 - 0.6747704)
    if Q.sum_z_dr2 > 0.004372139 and Q.planar_flow < 0.1950135:
        z += 2913.426 * (Q.sum_z_dr2 - 0.004372139) * (0.1950135 - Q.planar_flow)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.4447304
    if Q.sum_z_dr2 < 0.006679471:
        z += -165.4878 * Q.sum_z_dr2 + 1.105371
    if Q.tau1 < 0.05356915:
        z += -30.55927 * Q.tau1 + 1.637034
    if Q.LHA < 0.1967397:
        z += -12.13633 * Q.LHA + 2.387698
    if Q.log_sum_pt >= 6.701242:
        z += -7.011279 * Q.log_sum_pt + 46.98428
    if Q.sj3_dr_max < 0.1070199:
        z += 0.7796745 * Q.sj3_dr_max + 0.08035537
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += 12.66079 * Q.sj3_dr_max - 1.191161
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += -10.97014 * Q.sj3_dr_max + 2.178967
    if Q.sum_z_dr < 0.05464922:
        z += 49.68339 * Q.sum_z_dr - 2.715159
    if Q.zdr_5 < 0.005440034:
        z += -114.6925 * Q.zdr_5 + 0.6239311
    if Q.sum_z_dr2 < 0.006679471 and Q.D2_b2 < 4.721224:
        z += -42.05152 * (0.006679471 - Q.sum_z_dr2) * (4.721224 - Q.D2_b2)
    if Q.sum_z_dr2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 25184.43 * (0.006679471 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.sum_z_dr2 < 0.006679471 and Q.planar_flow < 0.4007947:
        z += 459.9938 * (0.006679471 - Q.sum_z_dr2) * (0.4007947 - Q.planar_flow)
    if Q.sum_z_dr2 < 0.005019719 and Q.centroid_offset > 0.006789738:
        z += -34770.57 * (0.005019719 - Q.sum_z_dr2) * (Q.centroid_offset - 0.006789738)
    if Q.LHA < 0.1967397 and Q.z_dr_0p2_0p4 < 0.20552:
        z += -182.1584 * (0.1967397 - Q.LHA) * (0.20552 - Q.z_dr_0p2_0p4)
    if Q.sum_z_dr2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += 3074.695 * (0.005019719 - Q.sum_z_dr2) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.tau1 < 0.05356915 and Q.sum_z_dr2 < 0.002635418:
        z += 35102.17 * (0.05356915 - Q.tau1) * (0.002635418 - Q.sum_z_dr2)
    if Q.sum_z_dr < 0.05464922 and Q.lam2 < 0.0001947983:
        z += 270570.9 * (0.05464922 - Q.sum_z_dr) * (0.0001947983 - Q.lam2)
    if Q.LHA < 0.1967397 and Q.lam2 < 0.0001947983:
        z += -80832.1 * (0.1967397 - Q.LHA) * (0.0001947983 - Q.lam2)
    if Q.log_sum_pt > 6.701242 and Q.pt_7 > 15.55391:
        z += -0.1503432 * (Q.log_sum_pt - 6.701242) * (Q.pt_7 - 15.55391)
    if Q.z_dr_0_0p05 > 0.9008535 and Q.lam2 < 0.0001947983:
        z += -55909.06 * (Q.z_dr_0_0p05 - 0.9008535) * (0.0001947983 - Q.lam2)
    if Q.sum_pt_top5 > 687.4375 and Q.e3 < 1.762929e-06:
        z += -6339.118 * (Q.sum_pt_top5 - 687.4375) * (1.762929e-06 - Q.e3)
    if Q.log_sum_pt > 6.701242 and Q.lam1_plus_lam2 < 0.002635418:
        z += 2880.616 * (Q.log_sum_pt - 6.701242) * (0.002635418 - Q.lam1_plus_lam2)
    if Q.sum_z_dr < 0.05464922 and Q.lam1_plus_lam2 < 0.002635418:
        z += -16099.0 * (0.05464922 - Q.sum_z_dr) * (0.002635418 - Q.lam1_plus_lam2)
    if Q.sum_z_dr < 0.05464922 and Q.lam1_plus_lam2 < 0.005590289:
        z += -11460.13 * (0.05464922 - Q.sum_z_dr) * (0.005590289 - Q.lam1_plus_lam2)
    if Q.sj3_dr_max < 0.1986272 and Q.lam2 < 0.0003061234:
        z += 25480.27 * (0.1986272 - Q.sj3_dr_max) * (0.0003061234 - Q.lam2)
    if Q.sj3_dr_max < 0.1070199 and Q.lam2 < 0.0003061234:
        z += -29443.64 * (0.1070199 - Q.sj3_dr_max) * (0.0003061234 - Q.lam2)
    return max(0.0, z)


def neuron_9(Q):
    z = -4.716725
    if Q.sum_z_dr < 0.05464922:
        z += 102.1577 * Q.sum_z_dr - 6.326545
    if 0.05464922 <= Q.sum_z_dr < 0.06663269:
        z += 62.06087 * Q.sum_z_dr - 4.135283
    if Q.tau1 < 0.04369778:
        z += 44.64406 * Q.tau1 - 1.950846
    if Q.sum_z_dr2 < 0.003562611:
        z += -2800.397 * Q.sum_z_dr2 + 11.76687
    if 0.003562611 <= Q.sum_z_dr2 < 0.005019719:
        z += -1228.556 * Q.sum_z_dr2 + 6.167006
    if Q.LHA < 0.2160559:
        z += -9.718805 * Q.LHA + 2.099805
    if Q.e3 >= 0.0001869378:
        z += 4787.092 * Q.e3 - 0.8948886
    if Q.centroid_offset < 0.01837778:
        z += -26.04805 * Q.centroid_offset + 1.427227
    if 0.01837778 <= Q.centroid_offset < 0.02355416:
        z += -183.2403 * Q.centroid_offset + 4.316071
    if Q.log_sum_pt < 6.896095:
        z += -6.862804 * Q.log_sum_pt + 47.32655
    if Q.lam1_plus_lam2 < 0.006679471:
        z += -907.5672 * Q.lam1_plus_lam2 + 6.062069
    if Q.lam2 >= 0.001130645:
        z += 1063.188 * Q.lam2 - 1.202087
    if Q.sj2_dr < 0.1294903:
        z += 25.17978 * Q.sj2_dr - 3.009844
    if 0.1294903 <= Q.sj2_dr < 0.1778793:
        z += -5.180787 * Q.sj2_dr + 0.9215547
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.5551115 * Q.n_dr_0p2_0p4 - 0.5551115
    if Q.sj3_dr_max < 0.1426152:
        z += 16.22223 * Q.sj3_dr_max - 1.513703
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += -14.27967 * Q.sj3_dr_max + 2.836331
    if Q.sj3_dr_max >= 0.2623172:
        z += -11.2415 * Q.sj3_dr_max + 2.948839
    if Q.sum_zz_dr2 < 0.0030133:
        z += 1421.79 * Q.sum_zz_dr2 - 5.217847
    if 0.0030133 <= Q.sum_zz_dr2 < 0.004641801:
        z += 573.2667 * Q.sum_zz_dr2 - 2.66099
    if Q.max_dr < 0.1117619:
        z += -24.81982 * Q.max_dr + 2.77391
    if Q.zdr_0 < 0.004918231:
        z += 281.2328 * Q.zdr_0 - 1.383168
    if Q.centroid_offset < 0.01837778 and Q.N3 < 1.395055:
        z += -319.2477 * (0.01837778 - Q.centroid_offset) * (1.395055 - Q.N3)
    if Q.centroid_offset < 0.01837778 and Q.tau3 < 0.01364517:
        z += 15774.05 * (0.01837778 - Q.centroid_offset) * (0.01364517 - Q.tau3)
    if Q.centroid_offset < 0.01837778 and Q.C2 < 0.06729223:
        z += 2164.145 * (0.01837778 - Q.centroid_offset) * (0.06729223 - Q.C2)
    if Q.lam2 > 0.001130645 and Q.planar_flow > 0.2534037:
        z += -999.2636 * (Q.lam2 - 0.001130645) * (Q.planar_flow - 0.2534037)
    if Q.log_sum_pt < 6.896095 and Q.z_5 < 0.05753583:
        z += 279.1841 * (6.896095 - Q.log_sum_pt) * (0.05753583 - Q.z_5)
    if Q.centroid_offset < 0.01837778 and Q.n_dr_0p05_0p1 < 5.0:
        z += 22.43585 * (0.01837778 - Q.centroid_offset) * (5.0 - Q.n_dr_0p05_0p1)
    if Q.centroid_offset < 0.02355416 and Q.tau21_b2 > 0.004811143:
        z += 156.2275 * (0.02355416 - Q.centroid_offset) * (Q.tau21_b2 - 0.004811143)
    if Q.centroid_offset < 0.02355416 and Q.D2_b2 < 0.380911:
        z += -418.7759 * (0.02355416 - Q.centroid_offset) * (0.380911 - Q.D2_b2)
    if Q.log_sum_pt < 6.896095 and Q.planar_flow > 0.1115136:
        z += -3.485909 * (6.896095 - Q.log_sum_pt) * (Q.planar_flow - 0.1115136)
    return max(0.0, z)


def neuron_10(Q):
    z = 2.394179
    if Q.sj3_dr_min >= 0.1278212:
        z += 11.58208 * Q.sj3_dr_min - 1.480435
    if Q.lam1 < 0.003377388:
        z += 1205.156 * Q.lam1 - 4.453598
    if 0.003377388 <= Q.lam1 < 0.00483998:
        z += 422.3682 * Q.lam1 - 1.809818
    if 0.00483998 <= Q.lam1 < 0.006506576:
        z += 257.9753 * Q.lam1 - 1.01416
    if Q.lam1 >= 0.006506576:
        z += -20.17694 * Q.lam1 + 0.7956588
    if 0.0001947983 <= Q.lam2 < 0.003408389:
        z += 1005.852 * Q.lam2 - 0.1959382
    if Q.lam2 >= 0.003408389:
        z += 291.7486 * Q.lam2 + 2.238004
    if Q.e3 >= 8.147744e-05:
        z += 3305.283 * Q.e3 - 0.269306
    if 0.1967397 <= Q.LHA < 0.3127275:
        z += -9.323752 * Q.LHA + 1.834352
    if Q.LHA >= 0.3127275:
        z += -28.4923 * Q.LHA + 7.828885
    if Q.n_dr_0p05_0p1 < 5.0:
        z += -0.1474252 * Q.n_dr_0p05_0p1 + 0.7371258
    if Q.tau1 >= 0.04369778:
        z += 20.38128 * Q.tau1 - 0.8906165
    if 0.1294903 <= Q.sj2_dr < 0.1682655:
        z += 10.7431 * Q.sj2_dr - 1.391128
    if Q.sj2_dr >= 0.1682655:
        z += -0.443717 * Q.sj2_dr + 0.4912289
    if Q.sum_z_dr2_top3 < 0.002915531:
        z += -302.933 * Q.sum_z_dr2_top3 + 0.8832105
    if Q.sum_z_dr2 >= 0.007520088:
        z += 292.5708 * Q.sum_z_dr2 - 2.200158
    if Q.lam2 > 0.0001947983 and Q.log_sum_pt < 6.538559:
        z += -819.9619 * (Q.lam2 - 0.0001947983) * (6.538559 - Q.log_sum_pt)
    if Q.lam2 > 0.0001947983 and Q.eccentricity > 0.4829602:
        z += -1970.939 * (Q.lam2 - 0.0001947983) * (Q.eccentricity - 0.4829602)
    if Q.lam2 > 0.0001947983 and Q.planar_flow > 0.06126205:
        z += -479.3072 * (Q.lam2 - 0.0001947983) * (Q.planar_flow - 0.06126205)
    if Q.sj2_dr > 0.1682655 and Q.planar_flow < 0.6947818:
        z += -30.97055 * (Q.sj2_dr - 0.1682655) * (0.6947818 - Q.planar_flow)
    if Q.LHA > 0.3127275 and Q.planar_flow < 0.6947818:
        z += -27.85783 * (Q.LHA - 0.3127275) * (0.6947818 - Q.planar_flow)
    return max(0.0, z)


def neuron_11(Q):
    z = -2.227369
    if Q.planar_flow < 0.2534037:
        z += -7.337837 * Q.planar_flow + 1.859435
    if 0.09419022 <= Q.sj2_dr < 0.1778793:
        z += 16.13066 * Q.sj2_dr - 1.51935
    if Q.sj2_dr >= 0.1778793:
        z += -8.232656 * Q.sj2_dr + 2.814379
    if Q.lam1_plus_lam2 < 0.008678045:
        z += -2117.232 * Q.lam1_plus_lam2 + 20.33507
    if 0.008678045 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -430.1224 * Q.lam1_plus_lam2 + 5.69425
    if Q.tau1 < 0.03459477:
        z += 56.9026 * Q.tau1 - 2.821842
    if 0.03459477 <= Q.tau1 < 0.09538712:
        z += 14.03646 * Q.tau1 - 1.338898
    if Q.LHA < 0.3033137:
        z += 2.343594 * Q.LHA - 1.133069
    if 0.3033137 <= Q.LHA < 0.3127275:
        z += 44.85211 * Q.LHA - 14.02649
    if Q.centroid_offset < 0.03776099:
        z += -62.63691 * Q.centroid_offset + 2.365232
    if Q.centroid_offset >= 0.04990367:
        z += 88.7111 * Q.centroid_offset - 4.427009
    if Q.lam1 < 0.00733008:
        z += 710.9072 * Q.lam1 - 6.066001
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 817.7911 * Q.lam1 - 6.849469
    if Q.sum_zz_dr2 < 0.006390125:
        z += 551.3949 * Q.sum_zz_dr2 - 4.083292
    if 0.006390125 <= Q.sum_zz_dr2 < 0.008168571:
        z += 314.7748 * Q.sum_zz_dr2 - 2.571261
    if Q.sum_z_dr2 < 0.003562611:
        z += 353.6909 * Q.sum_z_dr2 - 1.260063
    if Q.sum_z_dr < 0.0717028:
        z += 82.01492 * Q.sum_z_dr - 5.880699
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.01452835 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.planar_flow < 0.2534037 and Q.pt_7 < 37.15625:
        z += -0.3268414 * (0.2534037 - Q.planar_flow) * (37.15625 - Q.pt_7)
    if Q.sj2_dr > 0.1778793 and Q.lam2 < 0.001130645:
        z += -7485.988 * (Q.sj2_dr - 0.1778793) * (0.001130645 - Q.lam2)
    if Q.centroid_offset < 0.01437952 and Q.tau21_b2 < 0.05932655:
        z += 2831.374 * (0.01437952 - Q.centroid_offset) * (0.05932655 - Q.tau21_b2)
    if Q.centroid_offset < 0.01437952 and Q.sj3_dr_min < 0.08543881:
        z += -2079.477 * (0.01437952 - Q.centroid_offset) * (0.08543881 - Q.sj3_dr_min)
    if Q.tau1 < 0.03459477 and Q.sum_z_dr2_top3 < 0.002915531:
        z += 22588.71 * (0.03459477 - Q.tau1) * (0.002915531 - Q.sum_z_dr2_top3)
    if Q.sum_zz_dr2 < 0.0030133 and Q.sum_z_dr2_top3 > 0.003952582:
        z += -1017325.0 * (0.0030133 - Q.sum_zz_dr2) * (Q.sum_z_dr2_top3 - 0.003952582)
    if Q.pt_6 < 39.75 and Q.sum_z_dr2_top3 > 0.005011407:
        z += -14.11652 * (39.75 - Q.pt_6) * (Q.sum_z_dr2_top3 - 0.005011407)
    return max(0.0, z)


def neuron_12(Q):
    z = -1.362916
    if 0.01882765 <= Q.sum_z_dr2 < 0.02530566:
        z += 159.597 * Q.sum_z_dr2 - 3.004837
    if Q.sum_z_dr2 >= 0.02530566:
        z += 600.0669 * Q.sum_z_dr2 - 14.15122
    if Q.ptdr0_4 >= 15.26525:
        z += 0.2980273 * Q.ptdr0_4 - 4.549462
    if Q.ptdr0_4 > 15.26525 and Q.sum_pt < 988.4078:
        z += -0.0007659267 * (Q.ptdr0_4 - 15.26525) * (988.4078 - Q.sum_pt)
    if Q.sum_z_dr2 > 0.02530566 and Q.sum_pt < 868.5094:
        z += -0.8396523 * (Q.sum_z_dr2 - 0.02530566) * (868.5094 - Q.sum_pt)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.591964
    if Q.sum_z_dr < 0.1484084:
        z += -78.66021 * Q.sum_z_dr + 11.67384
    if Q.lam1 < 0.006506576:
        z += -419.3076 * Q.lam1 + 4.056252
    if 0.006506576 <= Q.lam1 < 0.01643375:
        z += -133.7737 * Q.lam1 + 2.198403
    if Q.z_7 >= 0.0586137:
        z += 48.55979 * Q.z_7 - 2.846269
    if Q.sj3_dr_max >= 0.233678:
        z += -14.39925 * Q.sj3_dr_max + 3.364787
    if Q.e2 < 0.08000524:
        z += 18.17356 * Q.e2 - 1.45398
    if Q.sj3_dr_min < 0.1278212:
        z += 13.3925 * Q.sj3_dr_min - 1.711844
    if Q.pt_7 >= 31.85938:
        z += -0.04552039 * Q.pt_7 + 1.450251
    if Q.tau1 < 0.1136369:
        z += 18.75102 * Q.tau1 - 2.130808
    if Q.lam1_plus_lam2 < 0.008678045:
        z += 148.4071 * Q.lam1_plus_lam2 - 0.7118401
    if 0.008678045 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -126.3078 * Q.lam1_plus_lam2 + 1.672148
    if Q.sum_z_dr < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -59.00831 * (0.1484084 - Q.sum_z_dr) * (6.804164 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.1484084 and Q.pt_7 < 38.53125:
        z += -0.9852244 * (0.1484084 - Q.sum_z_dr) * (38.53125 - Q.pt_7)
    if Q.sum_z_dr < 0.1484084 and Q.tau2 < 0.03585464:
        z += -297.7676 * (0.1484084 - Q.sum_z_dr) * (0.03585464 - Q.tau2)
    if Q.sum_pt_top5 > 687.4375 and Q.pt_7 < 40.04062:
        z += 0.0006727576 * (Q.sum_pt_top5 - 687.4375) * (40.04062 - Q.pt_7)
    if Q.lam1 < 0.01643375 and Q.pt_7 < 25.57812:
        z += -16.5999 * (0.01643375 - Q.lam1) * (25.57812 - Q.pt_7)
    if Q.e3 < 8.147744e-05 and Q.centroid_offset < 0.03776099:
        z += -684858.8 * (8.147744e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.sum_pt > 988.4078 and Q.sj3_pairmin_over_m > 0.2195798:
        z += -0.07408959 * (Q.sum_pt - 988.4078) * (Q.sj3_pairmin_over_m - 0.2195798)
    if Q.log_sum_pt < 6.46415 and Q.C3 < 0.06809619:
        z += -73.39734 * (6.46415 - Q.log_sum_pt) * (0.06809619 - Q.C3)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.5153412
    if 0.006679471 <= Q.sum_z_dr2 < 0.007520088:
        z += -1343.678 * Q.sum_z_dr2 + 8.975058
    if 0.007520088 <= Q.sum_z_dr2 < 0.008678045:
        z += -2111.334 * Q.sum_z_dr2 + 14.7479
    if Q.sum_z_dr2 >= 0.008678045:
        z += -1636.349 * Q.sum_z_dr2 + 10.62596
    if 0.002074109 <= Q.sum_zz_dr2 < 0.006390125:
        z += 171.8788 * Q.sum_zz_dr2 - 0.3564954
    if 0.006390125 <= Q.sum_zz_dr2 < 0.00718279:
        z += 1048.445 * Q.sum_zz_dr2 - 5.957865
    if 0.00718279 <= Q.sum_zz_dr2 < 0.008168571:
        z += 1934.509 * Q.sum_zz_dr2 - 12.32227
    if 0.008168571 <= Q.sum_zz_dr2 < 0.01716248:
        z += 1137.749 * Q.sum_zz_dr2 - 5.813885
    if Q.sum_zz_dr2 >= 0.01716248:
        z += 138.6055 * Q.sum_zz_dr2 + 11.3339
    if 0.005590289 <= Q.lam1_plus_lam2 < 0.01323868:
        z += 328.5797 * Q.lam1_plus_lam2 - 1.836856
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += 640.173 * Q.lam1_plus_lam2 - 5.961938
    if 0.03556091 <= Q.e2 < 0.04110972:
        z += -74.04343 * Q.e2 + 2.633052
    if Q.e2 >= 0.04110972:
        z += 3.039764 * Q.e2 - 0.5358163
    if 0.02355416 <= Q.centroid_offset < 0.04990367:
        z += -61.02752 * Q.centroid_offset + 1.437452
    if Q.centroid_offset >= 0.04990367:
        z += -268.4087 * Q.centroid_offset + 11.78653
    if 0.2160559 <= Q.LHA < 0.3033137:
        z += 11.59308 * Q.LHA - 2.504753
    if Q.LHA >= 0.3033137:
        z += 23.66992 * Q.LHA - 6.167824
    if 0.03459477 <= Q.tau1 < 0.09538712:
        z += -66.79971 * Q.tau1 + 2.310921
    if Q.tau1 >= 0.09538712:
        z += -51.94151 * Q.tau1 + 0.8936392
    if 0.02689598 <= Q.sum_z_dr < 0.04081947:
        z += 39.76181 * Q.sum_z_dr - 1.069433
    if 0.04081947 <= Q.sum_z_dr < 0.08723651:
        z += 141.2663 * Q.sum_z_dr - 5.212791
    if 0.08723651 <= Q.sum_z_dr < 0.1019409:
        z += -39.6121 * Q.sum_z_dr + 10.56641
    if Q.sum_z_dr >= 0.1019409:
        z += -198.0808 * Q.sum_z_dr + 26.72085
    if Q.sj3_dr_max < 0.1594012:
        z += -23.49577 * Q.sj3_dr_max + 0.1875114
    if 0.1594012 <= Q.sj3_dr_max < 0.213399:
        z += 51.57717 * Q.sj3_dr_max - 11.7792
    if 0.213399 <= Q.sj3_dr_max < 0.233678:
        z += 38.10277 * Q.sj3_dr_max - 8.903778
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -6385.152 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.01643375:
        z += 823.0002 * (0.1115136 - Q.planar_flow) * (0.01643375 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1_plus_lam2 < 0.00609665:
        z += 6717.39 * (0.1115136 - Q.planar_flow) * (0.00609665 - Q.lam1_plus_lam2)
    if Q.planar_flow < 0.1115136 and Q.centroid_offset < 0.01837778:
        z += -545.1481 * (0.1115136 - Q.planar_flow) * (0.01837778 - Q.centroid_offset)
    if Q.psi_0p1 > 0.8155839 and Q.centroid_offset > 0.03776099:
        z += -1531.551 * (Q.psi_0p1 - 0.8155839) * (Q.centroid_offset - 0.03776099)
    if Q.sum_z_dr2 > 0.008678045 and Q.log_sum_pt > 6.267538:
        z += -5128.79 * (Q.sum_z_dr2 - 0.008678045) * (Q.log_sum_pt - 6.267538)
    if Q.sum_zz_dr2 > 0.002074109 and Q.log_sum_pt > 6.267538:
        z += 440.8441 * (Q.sum_zz_dr2 - 0.002074109) * (Q.log_sum_pt - 6.267538)
    if Q.lam1_plus_lam2 > 0.01323868 and Q.sum_pt > 488.9312:
        z += 5.931192 * (Q.lam1_plus_lam2 - 0.01323868) * (Q.sum_pt - 488.9312)
    return max(0.0, z)


def neuron_15(Q):
    z = -1.709689
    if Q.sum_z_dr2 < 0.006679471:
        z += 1160.496 * Q.sum_z_dr2 - 4.87252
    if 0.006679471 <= Q.sum_z_dr2 < 0.007520088:
        z += 393.2406 * Q.sum_z_dr2 + 0.25234
    if 0.007520088 <= Q.sum_z_dr2 < 0.01323868:
        z += -561.2477 * Q.sum_z_dr2 + 7.430177
    if Q.sum_zz_dr2 < 0.005834489:
        z += -273.7254 * Q.sum_zz_dr2 + 1.458191
    if 0.005834489 <= Q.sum_zz_dr2 < 0.006390125:
        z += -570.9218 * Q.sum_zz_dr2 + 3.19218
    if 0.006390125 <= Q.sum_zz_dr2 < 0.01165737:
        z += 222.5486 * Q.sum_zz_dr2 - 1.878195
    if 0.01165737 <= Q.sum_zz_dr2 < 0.01716248:
        z += -130.0861 * Q.sum_zz_dr2 + 2.2326
    if Q.e2 < 0.04110972:
        z += -99.14261 * Q.e2 + 3.401757
    if 0.04110972 <= Q.e2 < 0.06344108:
        z += 30.18031 * Q.e2 - 1.914671
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += 844.5663 * Q.sum_z_dr2_top3 - 1.817142
    if Q.planar_flow < 0.1950135:
        z += 6.212458 * Q.planar_flow - 1.211513
    if Q.lam1 < 0.005433361:
        z += 39.60567 * Q.lam1 - 0.7316418
    if 0.005433361 <= Q.lam1 < 0.00595415:
        z += -177.6519 * Q.lam1 + 0.448797
    if 0.00595415 <= Q.lam1 < 0.008375572:
        z += 251.4924 * Q.lam1 - 2.106392
    if 0.1879486 <= Q.sj3_dr_max < 0.2623172:
        z += 11.18659 * Q.sj3_dr_max - 2.102504
    if Q.sj3_dr_max >= 0.2623172:
        z += 1.346339 * Q.sj3_dr_max + 0.4787643
    if Q.z_dr_0_0p05 < 0.1515405:
        z += -7.153549 * Q.z_dr_0_0p05 + 1.084052
    if Q.tau1 < 0.0811449:
        z += -29.43333 * Q.tau1 + 2.388365
    if Q.N2 < 0.2233283 and Q.sj2_dr < 0.1872617:
        z += -529.3256 * (0.2233283 - Q.N2) * (0.1872617 - Q.sj2_dr)
    if Q.N2 < 0.2233283 and Q.sum_z_dr2 > 0.008678045:
        z += -4732.213 * (0.2233283 - Q.N2) * (Q.sum_z_dr2 - 0.008678045)
    if Q.N2 < 0.2233283 and Q.LHA > 0.2931906:
        z += 429.1802 * (0.2233283 - Q.N2) * (Q.LHA - 0.2931906)
    if Q.N2 < 0.2233283 and Q.sj2_dr < 0.1591713:
        z += 393.3999 * (0.2233283 - Q.N2) * (0.1591713 - Q.sj2_dr)
    if Q.N2 < 0.2233283 and Q.LHA > 0.3255822:
        z += -293.3019 * (0.2233283 - Q.N2) * (Q.LHA - 0.3255822)
    if Q.N2 < 0.2233283 and Q.LHA > 0.3855647:
        z += -2454.06 * (0.2233283 - Q.N2) * (Q.LHA - 0.3855647)
    if Q.sum_z_dr2_top3 < 0.002151568 and Q.eccentricity > 0.9598562:
        z += -38007.99 * (0.002151568 - Q.sum_z_dr2_top3) * (Q.eccentricity - 0.9598562)
    if Q.sum_z_dr2 < 0.006679471 and Q.dr12 > 0.1587481:
        z += -86001.46 * (0.006679471 - Q.sum_z_dr2) * (Q.dr12 - 0.1587481)
    if Q.sum_zz_dr2 < 0.006390125 and Q.dr12 > 0.1587481:
        z += 80058.22 * (0.006390125 - Q.sum_zz_dr2) * (Q.dr12 - 0.1587481)
    if Q.planar_flow < 0.1950135 and Q.z_7 < 0.07148865:
        z += -233.3288 * (0.1950135 - Q.planar_flow) * (0.07148865 - Q.z_7)
    if Q.planar_flow < 0.1950135 and Q.sum_pt_top5 > 402.625:
        z += 0.06172876 * (0.1950135 - Q.planar_flow) * (Q.sum_pt_top5 - 402.625)
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
